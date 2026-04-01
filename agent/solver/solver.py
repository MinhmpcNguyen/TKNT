"""
Finite-pose bedroom/room layout solver using OR-Tools CP-SAT.

What this solver does
---------------------
- Builds generic mirrored variants for every cluster.
- Enumerates a finite set of hard-valid pose candidates for each cluster.
- Solves exact selection over that candidate set with CP-SAT:
  - exactly one pose per cluster
  - no pairwise overlap between selected poses
  - objective combines local pose quality + pairwise interaction bonuses
- Verifies the full layout.
- If the best exact solution is still not acceptable, refines only the most
  problematic clusters and solves again.

Important note
--------------
This is an exact solver over a finite candidate set, not a continuous global
optimizer over arbitrary x/y. It is more stable than an LLM search loop, but
still limited by the richness of the generated poses.

Expected companion file
-----------------------
This solver expects tools from cluster_placer/tools.py to be available, or
a path can be passed explicitly. That file provides robust geometry,
verification, canonicalization, mirroring, and candidate enumeration helpers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM

try:
    from ortools.sat.python import cp_model
except Exception as exc:  # pragma: no cover - import guard only
    raise RuntimeError(
        "OR-Tools is required. Install with: python -m pip install ortools"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    cluster_id: str
    variant_id: str
    variant_family: str
    variant_priority: float
    x: int
    y: int
    rot: int
    anchor_kind: str
    anchor_priority: float
    stage: str
    hard_valid: bool
    acceptable_valid: bool
    rough_score: int
    orientation_penalty_mm: int
    critical_orientation_penalty_mm: int
    focal_orientation_penalty_mm: int
    quality_gate_reasons: tuple[str, ...]
    hard_error_codes: tuple[str, ...]
    state_signature: str

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (self.cluster_id, self.variant_id, self.x, self.y, self.rot)


@dataclass(frozen=True)
class MacroClusterSolver:
    tools_path: str = ""
    max_variants_per_cluster: int = 8
    initial_candidates_per_cluster: int = 40
    max_rounds: int = 4
    time_limit_s: float = 20.0
    num_workers: int = 8

    def generate(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_outlines_json: dict[str, Any] | list[Any],
        relation_plan_json: dict[str, Any] | None,
        cluster_constraints_json: dict[str, Any] | None = None,
        grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    ) -> dict[str, Any]:
        tools_path = self.tools_path or _default_tools_path()
        return solve_layout(
            room_model=room_model_json,
            clusters_outlines=clusters_outlines_json,
            relation_plan=relation_plan_json,
            cluster_constraints=cluster_constraints_json,
            grid_mm=grid_mm,
            tools_path=tools_path,
            max_variants_per_cluster=self.max_variants_per_cluster,
            initial_candidates_per_cluster=self.initial_candidates_per_cluster,
            max_rounds=self.max_rounds,
            time_limit_s=self.time_limit_s,
            num_workers=self.num_workers,
        )


def _load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str | os.PathLike[str], data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_tools_module(path: str | os.PathLike[str]):
    spec = importlib.util.spec_from_file_location("layout_tools", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import tools module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_attr(mod: Any, name: str) -> Any:
    if not hasattr(mod, name):
        raise RuntimeError(f"tools module missing required symbol: {name}")
    return getattr(mod, name)


def _candidate_from_dict(item: dict[str, Any]) -> Candidate:
    return Candidate(
        cluster_id=str(item["cluster_id"]),
        variant_id=str(item["variant_id"]),
        variant_family=str(item.get("variant_family") or "base"),
        variant_priority=float(item.get("variant_priority") or 1.0),
        x=int(item["x"]),
        y=int(item["y"]),
        rot=int(item["rot"]),
        anchor_kind=str(item.get("anchor_kind") or "unknown"),
        anchor_priority=float(item.get("anchor_priority") or 0.0),
        stage=str(item.get("stage") or "seed"),
        hard_valid=bool(item.get("hard_valid", False)),
        acceptable_valid=bool(item.get("acceptable_valid", False)),
        rough_score=int(item.get("rough_score") or 0),
        orientation_penalty_mm=int(item.get("orientation_penalty_mm") or 0),
        critical_orientation_penalty_mm=int(
            item.get("critical_orientation_penalty_mm") or 0
        ),
        focal_orientation_penalty_mm=int(item.get("focal_orientation_penalty_mm") or 0),
        quality_gate_reasons=tuple(item.get("quality_gate_reasons") or []),
        hard_error_codes=tuple(item.get("hard_error_codes") or []),
        state_signature=str(item.get("state_signature") or ""),
    )


def _candidate_score(c: Candidate) -> int:
    score = int(c.rough_score)
    score += int(round(2200.0 * c.anchor_priority))
    score += int(round(1600.0 * c.variant_priority))
    if c.acceptable_valid:
        score += 4000
    score -= 3 * c.critical_orientation_penalty_mm
    score -= 2 * c.focal_orientation_penalty_mm
    score -= 1 * c.orientation_penalty_mm
    return score


def _canonicalize_inputs(
    tools: Any, room_model: Any, clusters_outlines: Any
) -> tuple[Any, Any]:
    unwrap = _require_attr(tools, "_unwrap_any")
    canon = _require_attr(tools, "_canonicalize_clusters_local_origin")
    room_u = unwrap(room_model)
    clusters_u = unwrap(clusters_outlines)
    clusters_c, _offsets = canon(clusters_u)
    return room_u, clusters_c


def _normalize_cluster_constraints(
    cluster_constraints: Any,
) -> dict[str, Any] | None:
    if not isinstance(cluster_constraints, dict):
        return None
    clusters = cluster_constraints.get("clusters")
    if not isinstance(clusters, list):
        return None
    out_clusters: list[dict[str, Any]] = []
    for row in clusters:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        out_clusters.append(row)
    if not out_clusters:
        return None
    return {"status": cluster_constraints.get("status"), "clusters": out_clusters}


def _cluster_constraints_summary(
    cluster_constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(cluster_constraints, dict):
        return {"cluster_count": 0, "access_object_count": 0}
    cluster_count = 0
    access_object_ids: set[tuple[str, str]] = set()
    for row in cluster_constraints.get("clusters") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        cluster_count += 1
        for item in row.get("hard_constraints") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() == "requires_access":
                oid = item.get("id")
                if isinstance(oid, str) and oid:
                    access_object_ids.add((cid, oid))
        rules = row.get("cluster_rules") or {}
        for item in rules.get("access_requirements") or []:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("required")):
                continue
            oid = item.get("id")
            if isinstance(oid, str) and oid:
                access_object_ids.add((cid, oid))
    return {
        "cluster_count": cluster_count,
        "access_object_count": len(access_object_ids),
    }


def _augment_relation_plan_from_cluster_constraints(
    relation_plan: Any,
    cluster_constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    import copy

    base = copy.deepcopy(relation_plan) if isinstance(relation_plan, dict) else {}
    if not isinstance(base.get("object_orientations"), list):
        base["object_orientations"] = []
    if not isinstance(base.get("notes"), list):
        base["notes"] = []

    existing = set()
    for item in base.get("object_orientations") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cluster_id") or "")
        oid = str(item.get("object_id") or "")
        intents = tuple(
            sorted(
                str(x).lower()
                for x in (item.get("intents") or [])
                if isinstance(x, str)
            )
        )
        existing.add((cid, oid, intents))

    if not isinstance(cluster_constraints, dict):
        return base

    added = 0
    for row in cluster_constraints.get("clusters") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue

        access_ids: set[str] = set()
        facing_ids: set[str] = set()

        for item in row.get("hard_constraints") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() == "requires_access":
                oid = item.get("id")
                mode = str(item.get("mode") or "").lower()
                if isinstance(oid, str) and oid and mode in {"front_clearance", ""}:
                    access_ids.add(oid)

        rules = row.get("cluster_rules") or {}
        for item in rules.get("access_requirements") or []:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("required")):
                continue
            oid = item.get("id")
            typ = str(item.get("type") or "").lower()
            if isinstance(oid, str) and oid and typ in {"front_clearance", ""}:
                access_ids.add(oid)

        facing = rules.get("facing") or {}
        if isinstance(facing, dict):
            for oid, meta in facing.items():
                if not isinstance(oid, str) or not oid:
                    continue
                if isinstance(meta, dict) and isinstance(meta.get("front"), str):
                    facing_ids.add(oid)

        # Access-bearing or explicitly fronted objects should expose their front to open space.
        for oid in sorted(access_ids | facing_ids):
            intents = tuple(sorted({"preserve_front_access", "front_to_open_space"}))
            sig = (cid, oid, intents)
            if sig in existing:
                continue
            base["object_orientations"].append(
                {
                    "cluster_id": cid,
                    "object_id": oid,
                    "intents": ["preserve_front_access", "front_to_open_space"],
                    "priority": "high" if oid in access_ids else "medium",
                    "reason": "augmented from cluster_constraints",
                }
            )
            existing.add(sig)
            added += 1

    if added > 0:
        base["notes"].append(
            f"augmented_object_orientations_from_cluster_constraints={added}"
        )
    return base


def _cluster_allowed_rotations(
    cluster_constraints: dict[str, Any] | None,
) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    if not isinstance(cluster_constraints, dict):
        return out
    for row in cluster_constraints.get("clusters") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        rules = row.get("cluster_rules") or {}
        allowed = rules.get("allowed_rotations")
        if not isinstance(allowed, dict):
            continue
        sets: list[set[int]] = []
        for _, vals in allowed.items():
            if not isinstance(vals, list):
                continue
            s = {int(v) % 360 for v in vals if int(v) % 90 == 0}
            if s:
                sets.append(s)
        if not sets:
            continue
        inter = set.intersection(*sets)
        if inter:
            out[cid] = inter
        else:
            uni = set.union(*sets)
            if uni:
                out[cid] = uni
    return out


def _filter_candidates_by_allowed_rotations(
    candidates_by_cluster: dict[str, list[Candidate]],
    cluster_allowed_rotations: dict[str, set[int]],
) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for cid, cands in candidates_by_cluster.items():
        allowed = cluster_allowed_rotations.get(cid)
        if not allowed:
            out[cid] = list(cands)
            continue
        out[cid] = [c for c in cands if int(c.rot) in allowed]
    return out


def _build_variant_payload_lookup(
    tools: Any, clusters_outlines: Any, *, max_variants_per_cluster: int
) -> dict[str, dict[str, dict[str, Any]]]:
    build_variants = _require_attr(tools, "BuildGenericClusterVariants")
    info = build_variants(
        clusters_outlines=clusters_outlines,
        max_variants_per_cluster=max_variants_per_cluster,
        include_variant_payloads=True,
    )
    if info.get("result") != "OK":
        raise RuntimeError(f"BuildGenericClusterVariants failed: {info}")
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for row in info.get("clusters") or []:
        cid = str(row.get("cluster_id") or "")
        vmap: dict[str, dict[str, Any]] = {}
        for v in row.get("variants") or []:
            payload = v.get("cluster_payload")
            vid = str(v.get("variant_id") or "")
            if vid and isinstance(payload, dict):
                vmap[vid] = payload
        if cid and vmap:
            lookup[cid] = vmap
    return lookup


def _enumerate_initial_candidates(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    max_candidates_per_cluster: int,
    max_variants_per_cluster: int,
) -> dict[str, list[Candidate]]:
    enum = _require_attr(tools, "EnumerateClusterCandidates")
    res = enum(
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        grid_mm=grid_mm,
        relation_plan=relation_plan,
        max_candidates_per_cluster=max_candidates_per_cluster,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    if res.get("result") != "OK":
        raise RuntimeError(f"EnumerateClusterCandidates failed: {res}")
    out: dict[str, list[Candidate]] = {}
    for row in res.get("clusters") or []:
        cid = str(row.get("cluster_id") or "")
        if not cid:
            continue
        out[cid] = [_candidate_from_dict(x) for x in (row.get("candidates") or [])]
    return out


def _relax_relation_plan_for_candidate_search(
    relation_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(relation_plan, dict):
        return relation_plan

    out = dict(relation_plan)
    out["cluster_relations"] = []
    out["cluster_directional_relations"] = []

    cluster_orientations: list[dict[str, Any]] = []
    for entry in relation_plan.get("cluster_orientations") or []:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        intents = [
            item
            for item in (entry.get("intents") or [])
            if isinstance(item, str) and item != "face_cluster"
        ]
        if not intents:
            continue
        new_entry["intents"] = intents
        if "face_cluster" not in intents:
            new_entry["target_cluster_id"] = None
        cluster_orientations.append(new_entry)
    out["cluster_orientations"] = cluster_orientations

    object_orientations: list[dict[str, Any]] = []
    for entry in relation_plan.get("object_orientations") or []:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        intents = [
            item
            for item in (entry.get("intents") or [])
            if isinstance(item, str)
            and item not in {"face_object", "face_away_from_object"}
        ]
        if not intents:
            continue
        new_entry["intents"] = intents
        if "face_object" not in intents and "face_away_from_object" not in intents:
            new_entry["target_object_id"] = None
        object_orientations.append(new_entry)
    out["object_orientations"] = object_orientations
    return out


def _augment_with_relaxed_candidates(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: dict[str, Any] | None,
    grid_mm: int,
    current_candidates: dict[str, list[Candidate]],
    cluster_allowed_rotations: dict[str, set[int]],
    max_variants_per_cluster: int,
    base_candidate_limit: int,
) -> dict[str, list[Candidate]]:
    relaxed_plan = _relax_relation_plan_for_candidate_search(relation_plan)
    extra_candidates = _enumerate_initial_candidates(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relaxed_plan,
        grid_mm=grid_mm,
        max_candidates_per_cluster=max(base_candidate_limit + 16, 56),
        max_variants_per_cluster=max_variants_per_cluster,
    )
    extra_candidates = _filter_candidates_by_allowed_rotations(
        extra_candidates,
        cluster_allowed_rotations,
    )

    merged: dict[str, list[Candidate]] = {}
    for cid in sorted(set(current_candidates.keys()) | set(extra_candidates.keys())):
        merged[cid] = _dedupe_candidates(
            list(current_candidates.get(cid, [])) + list(extra_candidates.get(cid, []))
        )[: max(base_candidate_limit + 32, 64)]
    return merged


def _dedupe_candidates(cands: Iterable[Candidate]) -> list[Candidate]:
    best: dict[tuple[str, str, int, int, int], Candidate] = {}
    for c in cands:
        prev = best.get(c.key)
        if prev is None or _candidate_score(c) > _candidate_score(prev):
            best[c.key] = c
    ordered = list(best.values())
    ordered.sort(key=lambda x: (-_candidate_score(x), x.variant_id, x.rot, x.x, x.y))
    return ordered


def _build_candidate_geometries(
    tools: Any,
    *,
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    candidates_by_cluster: dict[str, list[Candidate]],
) -> dict[tuple[str, int], Any]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    build_cluster_polys = _require_attr(tools, "_build_cluster_polys")
    fix_geom = _require_attr(tools, "_fix_geom")

    out: dict[tuple[str, int], Any] = {}
    for cid, cands in candidates_by_cluster.items():
        vmap = variant_payload_lookup.get(cid, {})
        for idx, cand in enumerate(cands):
            payload = vmap.get(cand.variant_id)
            if payload is None:
                raise RuntimeError(f"No payload for {cid} / {cand.variant_id}")
            polys = build_cluster_polys(Polygon, payload, cand.x, cand.y, cand.rot)
            if not polys:
                raise RuntimeError(f"No geometry for candidate {cand}")
            out[(cid, idx)] = fix_geom(unary_union(polys))
    return out


def _materialize_variantized_clusters(
    tools: Any,
    *,
    clusters_outlines: Any,
    selected_variants: dict[str, str],
) -> Any:
    materialize = _require_attr(tools, "MaterializeVariantizedClusters")
    result = materialize(
        clusters_outlines=clusters_outlines,
        selected_variants=selected_variants,
    )
    if result.get("result") != "OK":
        raise RuntimeError(f"MaterializeVariantizedClusters failed: {result}")
    return result.get("clusters_outlines")


def _pair_eval(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    a: Candidate,
    b: Candidate,
) -> dict[str, Any]:
    verify = _require_attr(tools, "GlobalClusterVerifier")
    selected_variants = {
        a.cluster_id: a.variant_id,
        b.cluster_id: b.variant_id,
    }
    mat = _materialize_variantized_clusters(
        tools,
        clusters_outlines=clusters_outlines,
        selected_variants=selected_variants,
    )
    return verify(
        room_model=room_model,
        clusters_outlines=mat,
        cluster_transforms=[
            {"cluster_id": a.cluster_id, "x": a.x, "y": a.y, "rot": a.rot},
            {"cluster_id": b.cluster_id, "x": b.x, "y": b.y, "rot": b.rot},
        ],
        grid_mm=grid_mm,
        mode="partial",
        relation_plan=relation_plan,
        return_debug=False,
    )


def _build_pair_terms(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    candidate_geoms: dict[tuple[str, int], Any],
    max_pair_bonus_abs: int = 12000,
) -> tuple[
    set[tuple[tuple[str, int], tuple[str, int]]],
    dict[tuple[tuple[str, int], tuple[str, int]], int],
]:
    incompatible: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    pair_bonus: dict[tuple[tuple[str, int], tuple[str, int]], int] = {}
    cluster_ids = sorted(candidates_by_cluster.keys())
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    for i, a_id in enumerate(cluster_ids):
        for j in range(i + 1, len(cluster_ids)):
            b_id = cluster_ids[j]
            a_cands = candidates_by_cluster[a_id]
            b_cands = candidates_by_cluster[b_id]
            for ai, ca in enumerate(a_cands):
                ga = candidate_geoms[(a_id, ai)]
                for bi, cb in enumerate(b_cands):
                    gb = candidate_geoms[(b_id, bi)]
                    key = ((a_id, ai), (b_id, bi))
                    try:
                        inter_area = float(ga.intersection(gb).area)
                    except Exception:
                        inter_area = 1.0
                    if inter_area > 1e-6:
                        incompatible.add(key)
                        continue

                    cache_key = (
                        a_id,
                        ai,
                        ca.variant_id,
                        ca.x,
                        ca.y,
                        ca.rot,
                        b_id,
                        bi,
                        cb.variant_id,
                        cb.x,
                        cb.y,
                        cb.rot,
                    )
                    pair_res = cache.get(cache_key)
                    if pair_res is None:
                        pair_res = _pair_eval(
                            tools,
                            room_model=room_model,
                            clusters_outlines=clusters_outlines,
                            relation_plan=relation_plan,
                            grid_mm=grid_mm,
                            a=ca,
                            b=cb,
                        )
                        cache[cache_key] = pair_res
                    if not bool(pair_res.get("hard_valid")):
                        incompatible.add(key)
                        continue
                    pair_layout_score = int(
                        ((pair_res.get("quality") or {}).get("layout_score") or 0)
                    )
                    bonus = (
                        pair_layout_score - _candidate_score(ca) - _candidate_score(cb)
                    )
                    bonus = int(
                        max(-max_pair_bonus_abs, min(max_pair_bonus_abs, bonus))
                    )
                    if bonus != 0:
                        pair_bonus[key] = bonus
    return incompatible, pair_bonus


def _selected_from_solver(
    candidates_by_cluster: dict[str, list[Candidate]],
    xvars: dict[tuple[str, int], cp_model.IntVar],
    solver: cp_model.CpSolver,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Candidate]]:
    transforms: list[dict[str, Any]] = []
    variants: list[dict[str, str]] = []
    chosen: dict[str, Candidate] = {}
    for cid, cands in candidates_by_cluster.items():
        found = None
        for i, cand in enumerate(cands):
            if solver.Value(xvars[(cid, i)]) == 1:
                found = cand
                break
        if found is None:
            raise RuntimeError(f"No selected candidate for {cid}")
        chosen[cid] = found
        transforms.append(
            {
                "cluster_id": cid,
                "x": found.x,
                "y": found.y,
                "rot": found.rot,
            }
        )
        variants.append({"cluster_id": cid, "variant_id": found.variant_id})
    transforms.sort(key=lambda x: x["cluster_id"])
    variants.sort(key=lambda x: x["cluster_id"])
    return transforms, variants, chosen


def _verify_complete(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    cluster_transforms: list[dict[str, Any]],
    selected_variants: list[dict[str, str]],
) -> dict[str, Any]:
    verify = _require_attr(tools, "GlobalClusterVerifier")
    mat = _materialize_variantized_clusters(
        tools,
        clusters_outlines=clusters_outlines,
        selected_variants={x["cluster_id"]: x["variant_id"] for x in selected_variants},
    )
    return verify(
        room_model=room_model,
        clusters_outlines=mat,
        cluster_transforms=cluster_transforms,
        grid_mm=grid_mm,
        mode="complete",
        relation_plan=relation_plan,
        return_debug=False,
    )


def _neighborhood_offsets(grid_mm: int) -> list[tuple[int, int]]:
    g = int(grid_mm)
    vals = [0, -g, g, -2 * g, 2 * g]
    out: list[tuple[int, int]] = []
    for dx in vals:
        for dy in vals:
            out.append((dx, dy))
    return out


def _expand_problem_clusters(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    chosen: dict[str, Candidate],
    problematic_cluster_ids: Sequence[str],
    max_variants_per_cluster: int,
    cluster_allowed_rotations: dict[str, set[int]] | None = None,
    add_limit_per_cluster: int = 36,
) -> dict[str, list[Candidate]]:
    verify = _require_attr(tools, "GlobalClusterVerifier")
    build_variants = _require_attr(tools, "BuildGenericClusterVariants")

    new_map = {cid: list(cands) for cid, cands in candidates_by_cluster.items()}
    variants_info = build_variants(
        clusters_outlines=clusters_outlines,
        cluster_ids=list(problematic_cluster_ids),
        max_variants_per_cluster=max_variants_per_cluster,
        include_variant_payloads=False,
    )
    variant_ids_by_cluster: dict[str, list[str]] = {}
    if variants_info.get("result") == "OK":
        for row in variants_info.get("clusters") or []:
            cid = str(row.get("cluster_id") or "")
            vids = [str(v.get("variant_id") or "") for v in (row.get("variants") or [])]
            variant_ids_by_cluster[cid] = [v for v in vids if v]

    for cid in problematic_cluster_ids:
        current = chosen.get(cid)
        if current is None:
            continue
        existing = {c.key for c in new_map.get(cid, [])}
        proposals: list[Candidate] = []
        variant_ids = variant_ids_by_cluster.get(cid) or [current.variant_id]
        allowed_rots = (cluster_allowed_rotations or {}).get(cid) or {0, 90, 180, 270}
        for variant_id in variant_ids[: min(4, len(variant_ids))]:
            for rot in tuple(r for r in (0, 90, 180, 270) if r in allowed_rots):
                for dx, dy in _neighborhood_offsets(grid_mm):
                    x = current.x + dx
                    y = current.y + dy
                    tmp = Candidate(
                        cluster_id=cid,
                        variant_id=variant_id,
                        variant_family=current.variant_family,
                        variant_priority=current.variant_priority,
                        x=x,
                        y=y,
                        rot=rot,
                        anchor_kind="local_refine",
                        anchor_priority=max(0.8, current.anchor_priority),
                        stage="refine_neighborhood",
                        hard_valid=False,
                        acceptable_valid=False,
                        rough_score=-(10**9),
                        orientation_penalty_mm=0,
                        critical_orientation_penalty_mm=0,
                        focal_orientation_penalty_mm=0,
                        quality_gate_reasons=(),
                        hard_error_codes=(),
                        state_signature="",
                    )
                    if tmp.key in existing:
                        continue
                    mat = _materialize_variantized_clusters(
                        tools,
                        clusters_outlines=clusters_outlines,
                        selected_variants={cid: variant_id},
                    )
                    res = verify(
                        room_model=room_model,
                        clusters_outlines=mat,
                        cluster_transforms=[
                            {"cluster_id": cid, "x": x, "y": y, "rot": rot}
                        ],
                        grid_mm=grid_mm,
                        mode="partial",
                        relation_plan=relation_plan,
                        return_debug=False,
                    )
                    if not bool(res.get("hard_valid")):
                        continue
                    q = res.get("quality") or {}
                    vrec = (res.get("violations_by_cluster") or {}).get(cid) or {}
                    candidate = Candidate(
                        cluster_id=cid,
                        variant_id=variant_id,
                        variant_family=current.variant_family,
                        variant_priority=current.variant_priority,
                        x=x,
                        y=y,
                        rot=rot,
                        anchor_kind="local_refine",
                        anchor_priority=max(0.8, current.anchor_priority),
                        stage="refine_neighborhood",
                        hard_valid=True,
                        acceptable_valid=True,
                        rough_score=int(q.get("layout_score") or 0),
                        orientation_penalty_mm=int(
                            vrec.get("orientation_penalty_mm") or 0
                        ),
                        critical_orientation_penalty_mm=int(
                            vrec.get("critical_orientation_penalty_mm") or 0
                        ),
                        focal_orientation_penalty_mm=int(
                            vrec.get("focal_orientation_penalty_mm") or 0
                        ),
                        quality_gate_reasons=tuple(
                            (res.get("quality_gate") or {}).get("reasons") or []
                        ),
                        hard_error_codes=tuple(),
                        state_signature=str(res.get("state_signature") or ""),
                    )
                    proposals.append(candidate)
                    existing.add(candidate.key)
                    if len(proposals) >= add_limit_per_cluster:
                        break
                if len(proposals) >= add_limit_per_cluster:
                    break
            if len(proposals) >= add_limit_per_cluster:
                break
        merged = _dedupe_candidates(list(new_map.get(cid, [])) + proposals)
        new_map[cid] = merged
    return new_map


def _solve_one_round(
    *,
    tools: Any,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    time_limit_s: float,
    num_workers: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], dict[str, Candidate], dict[str, Any]
]:
    candidate_geoms = _build_candidate_geometries(
        tools,
        variant_payload_lookup=variant_payload_lookup,
        candidates_by_cluster=candidates_by_cluster,
    )
    incompatible, pair_bonus = _build_pair_terms(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan,
        grid_mm=grid_mm,
        candidates_by_cluster=candidates_by_cluster,
        candidate_geoms=candidate_geoms,
    )

    model = cp_model.CpModel()
    xvars: dict[tuple[str, int], cp_model.IntVar] = {}
    for cid, cands in candidates_by_cluster.items():
        vars_for_cluster = []
        for i, cand in enumerate(cands):
            var = model.NewBoolVar(f"pick[{cid},{i}]")
            xvars[(cid, i)] = var
            vars_for_cluster.append(var)
        model.AddExactlyOne(vars_for_cluster)

    for a_key, b_key in sorted(incompatible):
        model.Add(xvars[a_key] + xvars[b_key] <= 1)

    objective_terms: list[Any] = []
    for cid, cands in candidates_by_cluster.items():
        for i, cand in enumerate(cands):
            objective_terms.append(_candidate_score(cand) * xvars[(cid, i)])

    for (a_key, b_key), bonus in pair_bonus.items():
        za = xvars[a_key]
        zb = xvars[b_key]
        z = model.NewBoolVar(f"pair[{a_key[0]},{a_key[1]}|{b_key[0]},{b_key[1]}]")
        model.Add(z <= za)
        model.Add(z <= zb)
        model.Add(z >= za + zb - 1)
        objective_terms.append(int(bonus) * z)

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(max(1, num_workers))
    solver.parameters.random_seed = 42

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT did not find a feasible assignment over the set")

    transforms, selected_variants, chosen = _selected_from_solver(
        candidates_by_cluster,
        xvars,
        solver,
    )
    verify = _verify_complete(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan,
        grid_mm=grid_mm,
        cluster_transforms=transforms,
        selected_variants=selected_variants,
    )
    return transforms, selected_variants, chosen, verify


def _filter_unsat_input_clusters(
    *,
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None,
) -> tuple[dict[str, Any] | list[Any], dict[str, Any] | None, dict[str, Any] | None]:
    filtered_outlines, removed_ids = _filter_unsat_clusters_outlines(clusters_outlines)
    if removed_ids:
        logger.info(
            "Solver filtered UNSAT clusters: %s",
            sorted(removed_ids),
        )
    if not removed_ids:
        return clusters_outlines, relation_plan, cluster_constraints
    filtered_relation_plan = _filter_relation_plan_clusters(relation_plan, removed_ids)
    filtered_constraints = _filter_cluster_constraints(cluster_constraints, removed_ids)
    return filtered_outlines, filtered_relation_plan, filtered_constraints


def _filter_unsat_clusters_outlines(
    clusters_outlines: dict[str, Any] | list[Any],
) -> tuple[dict[str, Any] | list[Any], set[str]]:
    removed_ids: set[str] = set()

    def is_unsat_status(value: Any) -> bool:
        return isinstance(value, str) and value.strip().upper() == "UNSAT"

    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            kept = []
            for item in clusters:
                if isinstance(item, dict) and is_unsat_status(item.get("status")):
                    cid = item.get("cluster_id")
                    if isinstance(cid, str) and cid:
                        removed_ids.add(cid)
                    continue
                kept.append(item)
            out = dict(clusters_outlines)
            out["clusters"] = kept
            return out, removed_ids

        out: dict[str, Any] = {}
        for cid, payload in clusters_outlines.items():
            if isinstance(payload, dict) and is_unsat_status(payload.get("status")):
                if isinstance(cid, str) and cid:
                    removed_ids.add(cid)
                continue
            out[cid] = payload
        return out, removed_ids

    if isinstance(clusters_outlines, list):
        kept_list: list[Any] = []
        for item in clusters_outlines:
            if isinstance(item, dict) and is_unsat_status(item.get("status")):
                cid = item.get("cluster_id")
                if isinstance(cid, str) and cid:
                    removed_ids.add(cid)
                continue
            kept_list.append(item)
        return kept_list, removed_ids

    return clusters_outlines, removed_ids


def _filter_relation_plan_clusters(
    relation_plan: dict[str, Any] | None, removed_ids: set[str]
) -> dict[str, Any] | None:
    if not removed_ids or not isinstance(relation_plan, dict):
        return relation_plan

    def _keep_cluster_id(value: Any) -> bool:
        return not (isinstance(value, str) and value in removed_ids)

    out = dict(relation_plan)

    def _filter_list(key: str, predicate) -> None:
        items = out.get(key)
        if not isinstance(items, list):
            return
        out[key] = [
            item for item in items if isinstance(item, dict) and predicate(item)
        ]

    _filter_list("cluster_affinities", lambda i: _keep_cluster_id(i.get("cluster_id")))
    _filter_list(
        "cluster_orientations",
        lambda i: _keep_cluster_id(i.get("cluster_id"))
        and _keep_cluster_id(i.get("target_cluster_id")),
    )
    _filter_list(
        "object_orientations",
        lambda i: _keep_cluster_id(i.get("cluster_id"))
        and _keep_cluster_id(i.get("target_cluster_id"))
        and _keep_cluster_id(i.get("target_object_cluster_id")),
    )
    _filter_list(
        "cluster_relations",
        lambda i: _keep_cluster_id(i.get("a")) and _keep_cluster_id(i.get("b")),
    )
    _filter_list(
        "cluster_directional_relations",
        lambda i: _keep_cluster_id(i.get("a")) and _keep_cluster_id(i.get("b")),
    )

    circ = out.get("circulation_plan")
    if isinstance(circ, dict):
        circ_out = dict(circ)
        main_paths = circ.get("main_paths")
        if isinstance(main_paths, list):
            circ_out["main_paths"] = [
                item
                for item in main_paths
                if isinstance(item, dict) and _keep_cluster_id(item.get("to_cluster"))
            ]
        keep_open = circ.get("keep_open_regions")
        if isinstance(keep_open, list):
            circ_out["keep_open_regions"] = [
                item
                for item in keep_open
                if isinstance(item, dict) and _keep_cluster_id(item.get("near"))
            ]
        out["circulation_plan"] = circ_out

    return out


def _filter_cluster_constraints(
    cluster_constraints: dict[str, Any] | None, removed_ids: set[str]
) -> dict[str, Any] | None:
    if not removed_ids or not isinstance(cluster_constraints, dict):
        return cluster_constraints
    clusters = cluster_constraints.get("clusters")
    if not isinstance(clusters, list):
        return cluster_constraints
    kept = [
        item
        for item in clusters
        if not (
            isinstance(item, dict)
            and isinstance(item.get("cluster_id"), str)
            and item.get("cluster_id") in removed_ids
        )
    ]
    out = dict(cluster_constraints)
    out["clusters"] = kept
    return out


def _filter_cluster_ids_from_outlines(
    clusters_outlines: dict[str, Any] | list[Any],
    removed_ids: set[str],
) -> dict[str, Any] | list[Any]:
    if not removed_ids:
        return clusters_outlines

    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            out = dict(clusters_outlines)
            out["clusters"] = [
                item
                for item in clusters
                if not (
                    isinstance(item, dict)
                    and isinstance(item.get("cluster_id"), str)
                    and item.get("cluster_id") in removed_ids
                )
            ]
            return out
        return {
            cid: payload
            for cid, payload in clusters_outlines.items()
            if not (isinstance(cid, str) and cid in removed_ids)
        }

    if isinstance(clusters_outlines, list):
        return [
            item
            for item in clusters_outlines
            if not (
                isinstance(item, dict)
                and isinstance(item.get("cluster_id"), str)
                and item.get("cluster_id") in removed_ids
            )
        ]

    return clusters_outlines


def _skip_clusters_without_candidates(
    *,
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None,
    candidates_by_cluster: dict[str, list[Candidate]],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    cluster_allowed_rotations: dict[str, set[int]],
) -> tuple[
    dict[str, Any] | list[Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, list[Candidate]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, set[int]],
    list[str],
]:
    skipped_ids = sorted(
        cid for cid, candidates in candidates_by_cluster.items() if not candidates
    )
    if not skipped_ids:
        return (
            clusters_outlines,
            relation_plan,
            cluster_constraints,
            candidates_by_cluster,
            variant_payload_lookup,
            cluster_allowed_rotations,
            [],
        )

    removed_ids = set(skipped_ids)
    filtered_outlines = _filter_cluster_ids_from_outlines(
        clusters_outlines, removed_ids
    )
    filtered_relation_plan = _filter_relation_plan_clusters(relation_plan, removed_ids)
    filtered_constraints = _filter_cluster_constraints(cluster_constraints, removed_ids)
    filtered_candidates = {
        cid: candidates
        for cid, candidates in candidates_by_cluster.items()
        if cid not in removed_ids
    }
    filtered_variants = {
        cid: variants
        for cid, variants in variant_payload_lookup.items()
        if cid not in removed_ids
    }
    filtered_rotations = {
        cid: rotations
        for cid, rotations in cluster_allowed_rotations.items()
        if cid not in removed_ids
    }
    return (
        filtered_outlines,
        filtered_relation_plan,
        filtered_constraints,
        filtered_candidates,
        filtered_variants,
        filtered_rotations,
        skipped_ids,
    )


def _selected_variants_map(
    selected_variants: Sequence[dict[str, str]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in selected_variants:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        vid = item.get("variant_id")
        if isinstance(cid, str) and cid and isinstance(vid, str) and vid:
            out[cid] = vid
    return out


def _build_placer_seed(
    tools: Any,
    *,
    clusters_outlines: Any,
    relation_plan_used: dict[str, Any] | None,
    cluster_constraints_used: dict[str, Any] | None,
    grid_mm: int,
    cluster_transforms: Sequence[dict[str, Any]],
    selected_variants: Sequence[dict[str, str]],
    verify: dict[str, Any] | None,
    solver_status: str,
    candidate_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not cluster_transforms or not selected_variants or not isinstance(verify, dict):
        return {
            "phase": "micro_repair",
            "ready": False,
            "needed": False,
            "reason": "no_complete_hard_valid_seed",
        }

    selected_variants_list = [
        {"cluster_id": str(x["cluster_id"]), "variant_id": str(x["variant_id"])}
        for x in selected_variants
        if isinstance(x, dict)
        and isinstance(x.get("cluster_id"), str)
        and isinstance(x.get("variant_id"), str)
    ]
    cluster_transforms_list = [
        {
            "cluster_id": str(x["cluster_id"]),
            "x": int(x["x"]),
            "y": int(x["y"]),
            "rot": int(x["rot"]),
        }
        for x in cluster_transforms
        if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
    ]

    materialized = _materialize_variantized_clusters(
        tools,
        clusters_outlines=clusters_outlines,
        selected_variants=_selected_variants_map(selected_variants_list),
    )
    quality = verify.get("quality") or {}
    quality_gate = verify.get("quality_gate") or {}
    repair_guidance = verify.get("repair_guidance") or {}
    acceptable = bool(verify.get("acceptable_valid"))

    return {
        "phase": "micro_repair",
        "ready": True,
        "needed": not acceptable,
        "seed_kind": "acceptable_valid" if acceptable else "hard_valid_complete",
        "solver_status": solver_status,
        "grid_mm": int(grid_mm),
        "seed_layout": {
            "cluster_transforms": cluster_transforms_list,
            "selected_variants": selected_variants_list,
        },
        "seed_metrics": {
            "layout_score": int(quality.get("layout_score") or 0),
            "state_signature": str(verify.get("state_signature") or ""),
            "hard_valid": bool(verify.get("hard_valid")),
            "acceptable_valid": acceptable,
            "quality_gate_reasons": list(quality_gate.get("reasons") or []),
        },
        "repair_targets": {
            "prioritized_clusters": list(
                repair_guidance.get("prioritized_clusters") or []
            ),
            "conflict_sets": list(repair_guidance.get("conflict_sets") or []),
            "top_critical_orientation_issues": list(
                (quality.get("critical_orientation_debug") or [])
            )[:16],
            "top_orientation_issues": list((quality.get("orientation_debug") or []))[
                :24
            ],
        },
        "edit_contract": {
            "must_preserve_hard_valid": True,
            "hard_constraints_must_hold": [
                "inside_room",
                "no_cluster_overlap",
                "no_hard_obstacle_intersection",
                "door_swing_clear",
            ],
            "primary_soft_targets": [
                "front_to_open_space",
                "preserve_front_access",
                "face_window",
                "critical_orientation_penalty",
                "focal_pair_penalty",
            ],
            "allowed_edit_levels": [
                "cluster_variant",
                "cluster_pose",
                "object_pose",
                "object_swap",
                "object_nudge",
            ],
        },
        "materialized_clusters_outlines": materialized,
        "relation_plan_used": relation_plan_used,
        "cluster_constraints_used": cluster_constraints_used,
        "seed_verify": verify,
        "candidate_counts": dict(candidate_counts or {}),
    }


def solve_layout(
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    cluster_constraints: Any | None = None,
    grid_mm: int,
    tools_path: str | os.PathLike[str],
    max_variants_per_cluster: int = 8,
    initial_candidates_per_cluster: int = 40,
    max_rounds: int = 4,
    time_limit_s: float = 20.0,
    num_workers: int = 8,
) -> dict[str, Any]:
    (
        clusters_outlines,
        relation_plan,
        cluster_constraints,
    ) = _filter_unsat_input_clusters(
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan if isinstance(relation_plan, dict) else None,
        cluster_constraints=(
            cluster_constraints if isinstance(cluster_constraints, dict) else None
        ),
    )

    tools = _load_tools_module(tools_path)

    room_model_c, clusters_c = _canonicalize_inputs(
        tools, room_model, clusters_outlines
    )
    cluster_constraints_n = _normalize_cluster_constraints(cluster_constraints)
    relation_plan_aug = _augment_relation_plan_from_cluster_constraints(
        relation_plan,
        cluster_constraints_n,
    )
    cluster_allowed_rots = _cluster_allowed_rotations(cluster_constraints_n)

    variant_payload_lookup = _build_variant_payload_lookup(
        tools,
        clusters_c,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    candidates_by_cluster = _enumerate_initial_candidates(
        tools,
        room_model=room_model_c,
        clusters_outlines=clusters_c,
        relation_plan=relation_plan_aug,
        grid_mm=grid_mm,
        max_candidates_per_cluster=initial_candidates_per_cluster,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    candidates_by_cluster = _filter_candidates_by_allowed_rotations(
        candidates_by_cluster,
        cluster_allowed_rots,
    )
    (
        clusters_c,
        relation_plan_aug,
        cluster_constraints_n,
        candidates_by_cluster,
        variant_payload_lookup,
        cluster_allowed_rots,
        skipped_candidate_clusters,
    ) = _skip_clusters_without_candidates(
        clusters_outlines=clusters_c,
        relation_plan=relation_plan_aug,
        cluster_constraints=cluster_constraints_n,
        candidates_by_cluster=candidates_by_cluster,
        variant_payload_lookup=variant_payload_lookup,
        cluster_allowed_rotations=cluster_allowed_rots,
    )
    if skipped_candidate_clusters:
        logger.info(
            "Solver skipped clusters without hard-valid candidates: %s",
            skipped_candidate_clusters,
        )
    cluster_ids = sorted(candidates_by_cluster.keys())
    if not cluster_ids:
        notes = ["No candidates were generated."]
        if skipped_candidate_clusters:
            notes.insert(
                0,
                f"Skipped clusters without hard-valid candidates: {skipped_candidate_clusters}",
            )
        return {
            "status": "UNSAT",
            "grid_mm": grid_mm,
            "cluster_transforms": [],
            "selected_variants": [],
            "notes": notes,
            "skipped_clusters": skipped_candidate_clusters,
            "placer_seed": {
                "phase": "micro_repair",
                "ready": False,
                "needed": False,
                "reason": "no_candidates_generated",
                "skipped_clusters": skipped_candidate_clusters,
            },
        }

    best_verify: dict[str, Any] | None = None
    best_transforms: list[dict[str, Any]] = []
    best_variants: list[dict[str, str]] = []
    best_score = -(10**18)
    used_relaxed_candidate_expansion = False
    infeasible_candidate_set = False

    current_candidates = {
        cid: list(vals) for cid, vals in candidates_by_cluster.items()
    }

    for round_idx in range(max_rounds):
        logger.info(
            "Solver round %s/%s: candidate_counts=%s",
            round_idx + 1,
            max_rounds,
            {cid: len(vals) for cid, vals in current_candidates.items()},
        )
        try:
            transforms, selected_variants, chosen, verify = _solve_one_round(
                tools=tools,
                room_model=room_model_c,
                clusters_outlines=clusters_c,
                relation_plan=relation_plan_aug,
                grid_mm=grid_mm,
                candidates_by_cluster=current_candidates,
                variant_payload_lookup=variant_payload_lookup,
                time_limit_s=time_limit_s,
                num_workers=num_workers,
            )
        except RuntimeError as exc:
            if not used_relaxed_candidate_expansion:
                logger.info(
                    "Solver exact solve was infeasible at round %s/%s; "
                    "expanding candidates with relaxed macro pairwise hints.",
                    round_idx + 1,
                    max_rounds,
                )
                current_candidates = _augment_with_relaxed_candidates(
                    tools,
                    room_model=room_model_c,
                    clusters_outlines=clusters_c,
                    relation_plan=relation_plan_aug,
                    grid_mm=grid_mm,
                    current_candidates=current_candidates,
                    cluster_allowed_rotations=cluster_allowed_rots,
                    max_variants_per_cluster=max_variants_per_cluster,
                    base_candidate_limit=initial_candidates_per_cluster,
                )
                used_relaxed_candidate_expansion = True
                continue
            logger.warning("Solver exact solve remained infeasible: %s", exc)
            infeasible_candidate_set = True
            break

        q = verify.get("quality") or {}
        score = int(q.get("layout_score") or 0)
        if best_verify is None or score > best_score:
            best_verify = verify
            best_score = score
            best_transforms = transforms
            best_variants = selected_variants

        if bool(verify.get("acceptable_valid")):
            logger.info(
                "Solver found acceptable layout at round %s/%s with layout_score=%s",
                round_idx + 1,
                max_rounds,
                score,
            )
            candidate_counts = {
                cid: len(vals) for cid, vals in current_candidates.items()
            }
            return {
                "status": "OK",
                "grid_mm": grid_mm,
                "cluster_transforms": transforms,
                "selected_variants": selected_variants,
                "notes": [
                    *(
                        [
                            "Skipped clusters without hard-valid candidates: "
                            f"{skipped_candidate_clusters}"
                        ]
                        if skipped_candidate_clusters
                        else []
                    ),
                    "Solved with finite-pose CP-SAT over enumerated candidates.",
                    f"layout_score={score}",
                    f"state_signature={verify.get('state_signature')}",
                    f"round={round_idx + 1}/{max_rounds}",
                ],
                "skipped_clusters": skipped_candidate_clusters,
                "placer_seed": _build_placer_seed(
                    tools,
                    clusters_outlines=clusters_c,
                    relation_plan_used=relation_plan_aug,
                    cluster_constraints_used=cluster_constraints_n,
                    grid_mm=grid_mm,
                    cluster_transforms=transforms,
                    selected_variants=selected_variants,
                    verify=verify,
                    solver_status="OK",
                    candidate_counts=candidate_counts,
                ),
                "solver_debug": {
                    "verify": verify,
                    "candidate_counts": candidate_counts,
                    "cluster_constraints_summary": _cluster_constraints_summary(
                        cluster_constraints_n
                    ),
                },
            }

        repair = verify.get("repair_guidance") or {}
        problem_clusters = [
            x.get("cluster_id")
            for x in (repair.get("prioritized_clusters") or [])
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        ]
        if not problem_clusters:
            problem_clusters = list(cluster_ids)

        logger.info(
            "Solver refine clusters at round %s/%s: %s",
            round_idx + 1,
            max_rounds,
            problem_clusters,
        )
        current_candidates = _expand_problem_clusters(
            tools,
            room_model=room_model_c,
            clusters_outlines=clusters_c,
            relation_plan=relation_plan_aug,
            grid_mm=grid_mm,
            candidates_by_cluster=current_candidates,
            chosen=chosen,
            problematic_cluster_ids=problem_clusters[: max(2, len(problem_clusters))],
            max_variants_per_cluster=max_variants_per_cluster,
            cluster_allowed_rotations=cluster_allowed_rots,
            add_limit_per_cluster=40,
        )
        # Keep the search exact but bounded.
        for cid, vals in current_candidates.items():
            current_candidates[cid] = _dedupe_candidates(vals)[
                : max(24, initial_candidates_per_cluster + 24)
            ]

    notes = [
        "No acceptable-valid solution found within the finite candidate space.",
    ]
    if infeasible_candidate_set:
        notes.insert(
            0,
            "No feasible exact assignment existed over the available macro candidate set.",
        )
    if used_relaxed_candidate_expansion:
        notes.append(
            "Expanded the candidate pool once with relaxed macro pairwise hints."
        )
    if best_verify is not None:
        reasons = (best_verify.get("quality_gate") or {}).get("reasons") or []
        notes.append(f"quality_gate_reasons={json.dumps(reasons, ensure_ascii=False)}")
        notes.append(f"state_signature={best_verify.get('state_signature')}")
        notes.append(f"best_layout_score={best_score}")
    candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
    return {
        "status": "UNSAT",
        "grid_mm": grid_mm,
        "cluster_transforms": [],
        "selected_variants": [],
        "notes": [
            *(
                [
                    "Skipped clusters without hard-valid candidates: "
                    f"{skipped_candidate_clusters}"
                ]
                if skipped_candidate_clusters
                else []
            ),
            *notes,
        ],
        "skipped_clusters": skipped_candidate_clusters,
        "placer_seed": _build_placer_seed(
            tools,
            clusters_outlines=clusters_c,
            relation_plan_used=relation_plan_aug,
            cluster_constraints_used=cluster_constraints_n,
            grid_mm=grid_mm,
            cluster_transforms=best_transforms,
            selected_variants=best_variants,
            verify=best_verify,
            solver_status="UNSAT",
            candidate_counts=candidate_counts,
        ),
        "solver_debug": {
            "best_verify": best_verify,
            "best_transforms": best_transforms,
            "best_selected_variants": best_variants,
            "candidate_counts": candidate_counts,
            "cluster_constraints_summary": _cluster_constraints_summary(
                cluster_constraints_n
            ),
        },
    }


def _default_tools_path() -> str:
    here = Path(__file__).resolve()
    return str(here.parents[2] / "cluster_placer" / "tools.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finite-pose room layout CP-SAT solver"
    )
    parser.add_argument("--room", required=True, help="Path to room_model JSON")
    parser.add_argument(
        "--clusters", required=True, help="Path to clusters_outlines JSON"
    )
    parser.add_argument("--relations", required=True, help="Path to relation_plan JSON")
    parser.add_argument(
        "--cluster-constraints",
        default=None,
        help="Optional path to cluster-level constraints JSON",
    )
    parser.add_argument(
        "--grid-mm",
        type=int,
        default=GLOBAL_LAYOUT_GRID_MM,
        help="Placement grid in mm",
    )
    parser.add_argument(
        "--tools", default=_default_tools_path(), help="Path to tools.py"
    )
    parser.add_argument("--max-variants", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--time-limit-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="solver_output.json")
    args = parser.parse_args(argv)

    room_model = _load_json(args.room)
    clusters = _load_json(args.clusters)
    relations = _load_json(args.relations)
    cluster_constraints = (
        _load_json(args.cluster_constraints) if args.cluster_constraints else None
    )

    result = solve_layout(
        room_model=room_model,
        clusters_outlines=clusters,
        relation_plan=relations,
        cluster_constraints=cluster_constraints,
        grid_mm=args.grid_mm,
        tools_path=args.tools,
        max_variants_per_cluster=args.max_variants,
        initial_candidates_per_cluster=args.max_candidates,
        max_rounds=args.rounds,
        time_limit_s=args.time_limit_s,
        num_workers=args.workers,
    )
    _dump_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
