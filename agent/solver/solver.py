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
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
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

_CLUSTER_WALL_PREFER_TAGS = frozenset(
    {"wall", "long_wall", "short_wall", "recess_or_edge"}
)
_CLUSTER_CENTER_PREFER_TAGS = frozenset({"center", "near_center"})
_CLUSTER_CENTER_AVOID_TAGS = frozenset({"center"})
_CLUSTER_WINDOW_AVOID_TAGS = frozenset({"window_blocking", "window_clearance"})
_CLUSTER_ENTRY_AVOID_TAGS = frozenset({"entry_blocking", "door_swing", "main_path"})
_CLUSTER_WINDOW_PREFER_TAGS = frozenset({"window_side"})
_CLUSTER_ENTRY_PREFER_TAGS = frozenset({"entry_side", "near_entry"})
_WALL_ANCHOR_TOKENS = ("wall", "top", "bottom", "left", "right", "edge", "recess")
_CENTER_ANCHOR_TOKENS = ("center", "quadrant")
_WINDOW_ANCHOR_TOKENS = ("window",)
_ENTRY_ANCHOR_TOKENS = ("entry", "door")

DEFAULT_GRID_MM = 50
DEFAULT_MAX_CONCEPTS = 5
DEFAULT_MAX_VARIANTS_PER_CLUSTER = 6
DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT = 40
DEFAULT_MAX_POSE_CANDIDATES_PER_CLUSTER = 120
DEFAULT_MAX_TOTAL_BINARY_CANDIDATES = 1200
DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT = 12
DEFAULT_MAX_CROSS_CONCEPT_FINALISTS = 20
DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT = 8.0
DEFAULT_SOLVER_TOTAL_TIME_LIMIT_SEC = 40.0
DEFAULT_MAX_DEGRADATION_ROUNDS = 3

QUALITY_WEIGHTS = {
    "functionality": 0.32,
    "naturalness": 0.24,
    "semantic": 0.24,
    "spatial": 0.20,
}

PUBLISHABLE_BLOCKING_QUALITY_REASONS = frozenset(
    {
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
        "critical_item_penalty_too_high",
    }
)

GENERIC_VARIANT_FAMILIES = frozenset(
    {
        "base",
        "mirror_x",
        "mirror_y",
        "mirror_xy",
        "front_flip",
        "front_flip_mirror_x",
        "front_flip_mirror_y",
        "composer",
        "variant",
    }
)

SEMANTIC_VARIANT_FAMILIES = frozenset(
    {
        "open_center",
        "perimeter_facing",
        "conversation_facing",
        "media_facing",
        "wall_backed_focal",
        "focal_media",
        "focal_axis",
        "focal_wall",
        "window_oriented",
        "daylight_work",
        "work_core",
        "workflow",
        "storage_wall",
        "edge_storage",
        "perimeter_storage",
        "support_edge",
        "fallback_generic",
    }
)

ROLE_VARIANT_FAMILY_ALLOWLISTS = {
    "social_anchor": frozenset(
        {
            "open_center",
            "perimeter_facing",
            "conversation_facing",
            "social_anchor",
            "social",
            "inward_facing",
        }
    ),
    "lounge": frozenset(
        {
            "open_center",
            "perimeter_facing",
            "conversation_facing",
            "social_anchor",
            "social",
            "inward_facing",
        }
    ),
    "media": frozenset(
        {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
            "focal_axis",
            "focal_wall",
            "media",
        }
    ),
    "focal": frozenset(
        {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
            "focal_axis",
            "focal_wall",
            "media",
        }
    ),
    "work": frozenset(
        {
            "daylight_work",
            "work_core",
            "window_oriented",
            "workflow",
            "workstation",
            "desk",
        }
    ),
    "workflow": frozenset(
        {
            "daylight_work",
            "work_core",
            "window_oriented",
            "workflow",
            "workstation",
            "desk",
        }
    ),
    "storage": frozenset(
        {
            "storage_wall",
            "wall_storage",
            "edge_storage",
            "perimeter_storage",
            "storage",
            "edge_weighted",
        }
    ),
}

STRICT_ROLE_KINDS = frozenset({"focal", "media", "work", "workflow"})


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
    macro_penalty_mm: int
    micro_penalty_mm: int
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
    max_variants_per_cluster: int = DEFAULT_MAX_VARIANTS_PER_CLUSTER
    initial_candidates_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT
    num_workers: int = 8
    max_feasible_solutions_per_concept: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT

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
            max_feasible_solutions_per_concept=self.max_feasible_solutions_per_concept,
        )

    def generate_bundle(
        self,
        payload: dict[str, Any],
        *,
        grid_mm: int = DEFAULT_GRID_MM,
    ) -> dict[str, Any]:
        tools_path = self.tools_path or _default_tools_path()
        return solve_global_layout_bundle(
            payload=payload,
            grid_mm=grid_mm,
            tools_path=tools_path,
            max_variants_per_cluster=self.max_variants_per_cluster,
            initial_candidates_per_cluster=self.initial_candidates_per_cluster,
            max_rounds=self.max_rounds,
            time_limit_s=self.time_limit_s,
            num_workers=self.num_workers,
            max_feasible_solutions_per_concept=self.max_feasible_solutions_per_concept,
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
        macro_penalty_mm=int(item.get("macro_penalty_mm") or 0),
        micro_penalty_mm=int(item.get("micro_penalty_mm") or 0),
        orientation_penalty_mm=int(item.get("orientation_penalty_mm") or 0),
        critical_orientation_penalty_mm=int(
            item.get("critical_orientation_penalty_mm") or 0
        ),
        focal_orientation_penalty_mm=int(item.get("focal_orientation_penalty_mm") or 0),
        quality_gate_reasons=tuple(item.get("quality_gate_reasons") or []),
        hard_error_codes=tuple(item.get("hard_error_codes") or []),
        state_signature=str(item.get("state_signature") or ""),
    )


def _primary_seed_problem_clusters(
    problem_clusters: list[str],
    *,
    cluster_ids: list[str],
    max_clusters: int = 2,
) -> list[str]:
    ordered: list[str] = []
    for cluster_id in problem_clusters or cluster_ids:
        clean_cluster_id = str(cluster_id or "").strip()
        if not clean_cluster_id or clean_cluster_id in ordered:
            continue
        ordered.append(clean_cluster_id)
        if len(ordered) >= max(1, int(max_clusters)):
            break
    return ordered


def _expanded_candidate_limit(initial_candidates_per_cluster: int) -> int:
    base_limit = max(12, int(initial_candidates_per_cluster))
    return max(24, base_limit + 8)


def _round_candidate_growth_limit(initial_candidates_per_cluster: int) -> int:
    base_limit = max(12, int(initial_candidates_per_cluster))
    return max(8, min(12, base_limit // 2))


def _candidate_score(c: Candidate) -> int:
    score = int(c.rough_score)
    score += int(round(2200.0 * c.anchor_priority))
    score += int(round(1600.0 * c.variant_priority))
    if c.hard_valid:
        score += 3200
    if c.acceptable_valid:
        score += 5200
    score -= 16 * c.macro_penalty_mm
    score -= 3 * c.micro_penalty_mm
    score -= 8 * c.critical_orientation_penalty_mm
    score -= 6 * c.focal_orientation_penalty_mm
    score -= 1 * c.orientation_penalty_mm
    return score


def _candidate_effective_score(
    c: Candidate,
    candidate_bonus_by_key: Mapping[tuple[str, str, int, int, int], int] | None = None,
) -> int:
    bonus = 0
    if candidate_bonus_by_key is not None:
        bonus = int(candidate_bonus_by_key.get(c.key, 0))
    return _candidate_score(c) + bonus


def _verify_quality_metrics(verify: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(verify, Mapping):
        return {
            "layout_score": 0,
            "macro_penalty_mm": 0,
            "micro_penalty_mm": 0,
            "critical_orientation_penalty_mm": 0,
            "focal_pair_penalty_mm": 0,
        }
    quality = verify.get("quality")
    summary = verify.get("summary")
    quality_dict = quality if isinstance(quality, Mapping) else {}
    summary_dict = summary if isinstance(summary, Mapping) else {}
    return {
        "layout_score": int(quality_dict.get("layout_score") or 0),
        "macro_penalty_mm": int(
            quality_dict.get("macro_penalty_mm")
            or summary_dict.get("macro_penalty_mm")
            or 0
        ),
        "micro_penalty_mm": int(
            quality_dict.get("micro_penalty_mm")
            or summary_dict.get("micro_penalty_mm")
            or 0
        ),
        "critical_orientation_penalty_mm": int(
            quality_dict.get("critical_orientation_penalty_mm")
            or summary_dict.get("critical_orientation_penalty_mm")
            or 0
        ),
        "focal_pair_penalty_mm": int(
            quality_dict.get("focal_pair_penalty_mm")
            or summary_dict.get("focal_pair_penalty_mm")
            or 0
        ),
    }


def _verify_search_score(verify: Mapping[str, Any] | None) -> int:
    metrics = _verify_quality_metrics(verify)
    score = int(metrics["layout_score"])
    if isinstance(verify, Mapping) and bool(verify.get("hard_valid")):
        score += 3200
    if isinstance(verify, Mapping) and bool(verify.get("acceptable_valid")):
        score += 5200
    score -= 16 * metrics["macro_penalty_mm"]
    score -= 3 * metrics["micro_penalty_mm"]
    score -= 8 * metrics["critical_orientation_penalty_mm"]
    score -= 6 * metrics["focal_pair_penalty_mm"]
    return score


def _verify_rank_key(verify: Mapping[str, Any] | None) -> tuple[int, ...]:
    metrics = _verify_quality_metrics(verify)
    hard_valid = (
        1 if isinstance(verify, Mapping) and bool(verify.get("hard_valid")) else 0
    )
    acceptable_valid = (
        1 if isinstance(verify, Mapping) and bool(verify.get("acceptable_valid")) else 0
    )
    complete = 1 if isinstance(verify, Mapping) and bool(verify.get("complete")) else 0
    return (
        acceptable_valid,
        hard_valid,
        complete,
        -metrics["critical_orientation_penalty_mm"],
        -metrics["focal_pair_penalty_mm"],
        -metrics["macro_penalty_mm"],
        -metrics["micro_penalty_mm"],
        metrics["layout_score"],
    )


def _verify_macro_ready(verify: Mapping[str, Any] | None) -> bool:
    if not isinstance(verify, Mapping):
        return False
    if not bool(verify.get("hard_valid")) or not bool(verify.get("complete")):
        return False
    quality_gate = verify.get("quality_gate")
    reasons = []
    if isinstance(quality_gate, Mapping):
        reasons = [str(item or "") for item in (quality_gate.get("reasons") or [])]
    blocking = {
        "hard_constraints_failed",
        "layout_incomplete",
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
    }
    return not any(reason in blocking for reason in reasons)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _penalty_score(penalty_mm: Any, scale_mm: float) -> float:
    penalty = max(0.0, _as_float(penalty_mm))
    return _clamp_score(1.0 / (1.0 + (penalty / max(1.0, scale_mm))))


def _quality_dict(verify: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(verify, Mapping):
        return {}
    quality = verify.get("quality")
    return quality if isinstance(quality, Mapping) else {}


def _summary_dict(verify: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(verify, Mapping):
        return {}
    summary = verify.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def _concept_from_relation_plan(
    relation_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    concept = relation_plan.get("macro_concept")
    return concept if isinstance(concept, Mapping) else {}


def _concept_id_from_relation_plan(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    for key in ("concept_id", "id", "concept_family"):
        value = str(concept.get(key) or "").strip()
        if value:
            return value
    if isinstance(relation_plan, Mapping):
        value = str(relation_plan.get("concept_id") or "").strip()
        if value:
            return value
    return "concept_01"


def _concept_prior_score(relation_plan: Mapping[str, Any] | None) -> float:
    concept = _concept_from_relation_plan(relation_plan)
    value = _as_float(concept.get("concept_score_prior"), 0.78)
    if value > 1.0:
        value /= 100.0
    return _clamp_score(value)


def _center_openness_strength(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    if isinstance(topology_policy, Mapping):
        strength = str(topology_policy.get("reserve_center_degree") or "").strip()
        if strength:
            return strength
    if isinstance(relation_plan, Mapping):
        intent = relation_plan.get("layout_intent_profile")
        if isinstance(intent, Mapping):
            return str(intent.get("center_open_preference") or "").strip()
    return ""


def _normalized_policy_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _concept_family_from_relation_plan(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    return _normalized_policy_token(concept.get("concept_family"))


def _center_openness_is_strong(relation_plan: Mapping[str, Any] | None) -> bool:
    return _normalized_policy_token(_center_openness_strength(relation_plan)) in {
        "high",
        "very_high",
    }


def _center_openness_is_very_high(relation_plan: Mapping[str, Any] | None) -> bool:
    return _normalized_policy_token(_center_openness_strength(relation_plan)) == (
        "very_high"
    )


def _cluster_zone_plan_by_id(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    concept = _concept_from_relation_plan(relation_plan)
    zone_plan = concept.get("cluster_zone_plan")
    if not isinstance(zone_plan, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for row in zone_plan:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        if cluster_id:
            out[cluster_id] = row
    return out


def _cluster_role_kind(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> str:
    row = _cluster_zone_plan_by_id(relation_plan).get(cluster_id, {})
    explicit = _normalized_policy_token(
        row.get("role_kind")
        or row.get("cluster_role_kind")
        or row.get("role")
        or row.get("cluster_role")
    )
    if explicit:
        return explicit
    tokens = " ".join(
        str(value or "")
        for value in (
            cluster_id,
            row.get("semantic_role"),
            row.get("zone_assignment"),
            row.get("priority"),
        )
    ).lower()
    if any(token in tokens for token in ("sofa", "sectional", "lounge", "seating")):
        return "social_anchor"
    if any(token in tokens for token in ("tv", "media", "screen", "fireplace")):
        return "media"
    if any(token in tokens for token in ("desk", "work", "office", "study")):
        return "work"
    if any(token in tokens for token in ("storage", "cabinet", "shelf", "wardrobe")):
        return "storage"
    if any(token in tokens for token in ("focal", "feature_wall")):
        return "focal"
    return "support"


def _cluster_priority_kind(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> str:
    row = _cluster_zone_plan_by_id(relation_plan).get(cluster_id, {})
    priority = _normalized_policy_token(row.get("priority"))
    if priority in {"core", "support", "optional"}:
        return priority
    role_kind = _cluster_role_kind(relation_plan, cluster_id)
    if role_kind in {"social_anchor", "media", "focal", "sleep"}:
        return "core"
    if role_kind in {"work", "storage"}:
        return "support"
    return "support"


def _cluster_is_core_or_protected(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> bool:
    if cluster_id in {
        _extract_layout_primary_cluster_id(
            dict(relation_plan) if isinstance(relation_plan, dict) else None
        ),
        _extract_layout_secondary_cluster_id(
            dict(relation_plan) if isinstance(relation_plan, dict) else None
        ),
    }:
        return True
    return _cluster_priority_kind(relation_plan, cluster_id) == "core"


def _explicit_allowed_variant_families_by_cluster(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    containers: list[Any] = []
    if isinstance(relation_plan, Mapping):
        containers.append(relation_plan.get("allowed_variant_families_by_cluster"))
        policy = relation_plan.get("variant_family_policy")
        if isinstance(policy, Mapping):
            containers.append(policy.get("allowed_variant_families_by_cluster"))
            containers.append(policy.get("allowed_by_cluster"))
    concept = _concept_from_relation_plan(relation_plan)
    containers.append(concept.get("allowed_variant_families_by_cluster"))
    policy = concept.get("variant_family_policy")
    if isinstance(policy, Mapping):
        containers.append(policy.get("allowed_variant_families_by_cluster"))
        containers.append(policy.get("allowed_by_cluster"))

    out: dict[str, set[str]] = {}
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for cluster_id_raw, raw_value in container.items():
            cluster_id = str(cluster_id_raw or "").strip()
            if not cluster_id:
                continue
            values: Any
            if isinstance(raw_value, Mapping):
                values = (
                    raw_value.get("families")
                    or raw_value.get("allowed")
                    or raw_value.get("variant_families")
                )
            else:
                values = raw_value
            if isinstance(values, str):
                families = {_normalized_policy_token(values)}
            elif isinstance(values, Sequence):
                families = {
                    _normalized_policy_token(value)
                    for value in values
                    if _normalized_policy_token(value)
                }
            else:
                families = set()
            if families:
                out.setdefault(cluster_id, set()).update(families)
    return out


def _concept_allows_fallback_generic(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> bool:
    explicit = _explicit_allowed_variant_families_by_cluster(relation_plan).get(
        cluster_id,
        set(),
    )
    if "fallback_generic" in explicit:
        return True
    concept = _concept_from_relation_plan(relation_plan)
    for key in (
        "allow_fallback_generic",
        "fallback_generic_allowed",
        "allow_generic_fallback",
    ):
        value = concept.get(key)
        if isinstance(value, bool):
            return value
    policy = concept.get("variant_family_policy")
    if isinstance(policy, Mapping):
        for key in (
            "allow_fallback_generic",
            "fallback_generic_allowed",
            "allow_generic_fallback",
        ):
            value = policy.get(key)
            if isinstance(value, bool):
                return value
    return False


def _role_allowed_variant_families(role_kind: str) -> set[str]:
    normalized = _normalized_policy_token(role_kind)
    return set(ROLE_VARIANT_FAMILY_ALLOWLISTS.get(normalized, frozenset()))


def _variant_family_compatibility_score(
    *,
    concept_family: str,
    role_kind: str,
    variant_family: str,
) -> float:
    concept = _normalized_policy_token(concept_family)
    role = _normalized_policy_token(role_kind)
    family = _normalized_policy_token(variant_family)
    if family == "fallback_generic":
        return -1.2 if role in {"social_anchor", "media", "focal", "work"} else -0.7
    if family in GENERIC_VARIANT_FAMILIES:
        return 0.0

    role_allowed = _role_allowed_variant_families(role)
    score = 0.0
    if role_allowed:
        score += 0.75 if family in role_allowed else -0.65

    if concept == "open_center":
        if role in {"social_anchor", "lounge"} and family in {
            "open_center",
            "perimeter_facing",
            "conversation_facing",
        }:
            score += 0.8
        elif role in {"media", "focal"} and family in {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
        }:
            score += 0.45
        elif family in {"storage_wall", "edge_storage", "perimeter_storage"}:
            score += 0.25
    elif concept == "focal_axis":
        if role in {"media", "focal"} and family in {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
            "focal_axis",
        }:
            score += 0.9
        elif role in {"social_anchor", "lounge"} and family in {
            "conversation_facing",
            "perimeter_facing",
        }:
            score += 0.35
    elif concept == "daylight_oriented":
        if role in {"work", "workflow"} and family in {
            "daylight_work",
            "work_core",
            "window_oriented",
        }:
            score += 0.95
        elif family == "window_oriented":
            score += 0.35
    elif concept == "edge_weighted":
        if family in {"storage_wall", "edge_storage", "perimeter_storage"}:
            score += 0.65
        elif family in {"open_center", "conversation_facing"}:
            score -= 0.25
    elif concept == "zoned":
        if role_allowed and family in role_allowed:
            score += 0.35
    return max(-1.5, min(1.5, score))


def _variant_family_allowed_for_cluster(
    *,
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
    variant_family: str,
    semantic_families_available: bool,
) -> bool:
    family = _normalized_policy_token(variant_family)
    if not family:
        return True
    if family == "fallback_generic":
        return _concept_allows_fallback_generic(relation_plan, cluster_id)

    explicit = _explicit_allowed_variant_families_by_cluster(relation_plan).get(
        cluster_id,
        set(),
    )
    if explicit:
        return family in explicit

    role_kind = _cluster_role_kind(relation_plan, cluster_id)
    role_allowed = _role_allowed_variant_families(role_kind)
    if role_allowed and family in role_allowed:
        return True
    if _normalized_policy_token(role_kind) in STRICT_ROLE_KINDS:
        return False
    if family in GENERIC_VARIANT_FAMILIES:
        return not semantic_families_available
    if role_allowed:
        return False
    return True


def _filter_candidates_by_concept_variant_policy(
    candidates_by_cluster: dict[str, list[Candidate]],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for cluster_id, candidates in candidates_by_cluster.items():
        semantic_families_available = any(
            _normalized_policy_token(candidate.variant_family)
            in SEMANTIC_VARIANT_FAMILIES
            for candidate in candidates
        )
        out[cluster_id] = [
            candidate
            for candidate in candidates
            if _variant_family_allowed_for_cluster(
                relation_plan=relation_plan,
                cluster_id=cluster_id,
                variant_family=candidate.variant_family,
                semantic_families_available=semantic_families_available,
            )
        ]
    return out


def _filter_variant_bundle_rows_for_concept(
    *,
    cluster_id: str,
    rows: Any,
    relation_plan: Mapping[str, Any] | None,
) -> Any:
    if not isinstance(rows, list):
        return rows
    variant_rows = [row for row in rows if isinstance(row, Mapping)]
    if not variant_rows:
        return rows
    semantic_families_available = any(
        _normalized_policy_token(row.get("variant_family")) in SEMANTIC_VARIANT_FAMILIES
        for row in variant_rows
    )
    filtered = [
        dict(row)
        for row in variant_rows
        if _variant_family_allowed_for_cluster(
            relation_plan=relation_plan,
            cluster_id=cluster_id,
            variant_family=str(row.get("variant_family") or ""),
            semantic_families_available=semantic_families_available,
        )
    ]
    return filtered or rows


def _filter_clusters_outlines_by_concept_variant_policy(
    clusters_outlines: Any,
    relation_plan: Mapping[str, Any] | None,
) -> Any:
    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            out = dict(clusters_outlines)
            out_clusters = []
            for item in clusters:
                if not isinstance(item, dict):
                    out_clusters.append(item)
                    continue
                cluster_id = str(item.get("cluster_id") or "").strip()
                next_item = dict(item)
                next_item["variant_bundle"] = _filter_variant_bundle_rows_for_concept(
                    cluster_id=cluster_id,
                    rows=item.get("variant_bundle"),
                    relation_plan=relation_plan,
                )
                out_clusters.append(next_item)
            out["clusters"] = out_clusters
            return out

        out: dict[str, Any] = {}
        for cluster_id, payload in clusters_outlines.items():
            if not isinstance(payload, dict):
                out[cluster_id] = payload
                continue
            next_payload = dict(payload)
            next_payload["variant_bundle"] = _filter_variant_bundle_rows_for_concept(
                cluster_id=str(cluster_id),
                rows=payload.get("variant_bundle"),
                relation_plan=relation_plan,
            )
            out[cluster_id] = next_payload
        return out

    if isinstance(clusters_outlines, list):
        out_list = []
        for item in clusters_outlines:
            if not isinstance(item, dict):
                out_list.append(item)
                continue
            cluster_id = str(item.get("cluster_id") or "").strip()
            next_item = dict(item)
            next_item["variant_bundle"] = _filter_variant_bundle_rows_for_concept(
                cluster_id=cluster_id,
                rows=item.get("variant_bundle"),
                relation_plan=relation_plan,
            )
            out_list.append(next_item)
        return out_list

    return clusters_outlines


def _filter_protected_topology_candidates(
    candidates_by_cluster: dict[str, list[Candidate]],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, list[Candidate]]:
    if not _center_openness_is_strong(relation_plan):
        return candidates_by_cluster
    profiles, _adjacency = _build_cluster_macro_profiles(
        dict(relation_plan) if isinstance(relation_plan, dict) else None,
        cluster_ids=list(candidates_by_cluster.keys()),
    )
    out: dict[str, list[Candidate]] = {}
    for cluster_id, candidates in candidates_by_cluster.items():
        profile = profiles.get(cluster_id, {})
        filtered = list(candidates)
        if _cluster_avoids_center(profile):
            non_center = [
                candidate
                for candidate in filtered
                if not _anchor_kind_is_center(candidate.anchor_kind)
            ]
            if non_center:
                filtered = non_center
        if _cluster_avoids_entry(profile):
            non_entry = [
                candidate
                for candidate in filtered
                if not _anchor_kind_is_entry(candidate.anchor_kind)
            ]
            if non_entry:
                filtered = non_entry
        out[cluster_id] = filtered
    return out


def _solution_signature(
    cluster_transforms: Sequence[dict[str, Any]],
    selected_variants: Sequence[dict[str, str]],
) -> str:
    variant_pairs = sorted(
        (
            str(item.get("cluster_id") or ""),
            str(item.get("variant_id") or ""),
        )
        for item in selected_variants
        if isinstance(item, dict)
    )
    transform_pairs = sorted(
        (
            str(item.get("cluster_id") or ""),
            _as_int(item.get("x")),
            _as_int(item.get("y")),
            _as_int(item.get("rot")) % 360,
        )
        for item in cluster_transforms
        if isinstance(item, dict)
    )
    return json.dumps(
        {"variants": variant_pairs, "transforms": transform_pairs},
        sort_keys=True,
        separators=(",", ":"),
    )


def _dominant_anchor_correct(
    *,
    relation_plan: Mapping[str, Any] | None,
    chosen: Mapping[str, Candidate],
) -> bool:
    primary_cluster_id = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if primary_cluster_id is None:
        return True
    candidate = chosen.get(primary_cluster_id)
    if candidate is None:
        return False
    profiles, _adjacency = _build_cluster_macro_profiles(
        dict(relation_plan) if isinstance(relation_plan, dict) else None,
        cluster_ids=[primary_cluster_id],
    )
    profile = profiles.get(primary_cluster_id, {})
    if _cluster_prefers_wall(profile):
        return _anchor_kind_is_wall_pinned(candidate.anchor_kind)
    if _cluster_prefers_center(profile):
        return _anchor_kind_is_center(candidate.anchor_kind)
    return True


def _cluster_rows_from_verify(
    verify: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(verify, Mapping):
        return {}
    rows_src = None
    materialized = verify.get("materialized_layout")
    if isinstance(materialized, Mapping):
        rows_src = materialized.get("clusters")
    if rows_src is None:
        rows_src = verify.get("clusters")
    if not isinstance(rows_src, Sequence) or isinstance(rows_src, str):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows_src:
        if not isinstance(row, Mapping):
            continue
        cid = str(row.get("cluster_id") or "").strip()
        if cid:
            out[cid] = row
    return out


def _vector_tuple(value: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    dx = value.get("dx")
    dy = value.get("dy")
    try:
        x = float(dx)
        y = float(dy)
    except (TypeError, ValueError):
        return None
    norm = math.hypot(x, y)
    if norm <= 1e-6:
        return None
    return (x / norm, y / norm)


def _cluster_center_tuple(row: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(row, Mapping):
        return None
    center = row.get("world_center")
    if isinstance(center, Mapping):
        try:
            return (float(center.get("x")), float(center.get("y")))
        except (TypeError, ValueError):
            return None
    bbox = row.get("bbox")
    if isinstance(bbox, Mapping):
        try:
            min_x = float(bbox.get("min_x"))
            max_x = float(bbox.get("max_x"))
            min_y = float(bbox.get("min_y"))
            max_y = float(bbox.get("max_y"))
            return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        except (TypeError, ValueError):
            return None
    return None


def _normalized_direction(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float] | None:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return (dx / norm, dy / norm)


def _dot(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(a[0] * b[0] + a[1] * b[1])


def _viewing_pairs_from_relation_plan(
    relation_plan: Mapping[str, Any] | None,
) -> list[tuple[str, str, str]]:
    if not isinstance(relation_plan, Mapping):
        return []
    pairs: list[tuple[str, str, str]] = []
    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, Mapping):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"face_each_other", "access_faces_other"}:
            continue
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if a and b:
            pairs.append((a, b, relation))
    layout_intent = relation_plan.get("layout_intent_profile") or {}
    primary = str(layout_intent.get("primary_cluster_id") or "").strip()
    secondary = str(layout_intent.get("secondary_cluster_id") or "").strip()
    focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
    if primary and secondary and focus_mode in {"viewing", "mixed"}:
        if not any({primary, secondary} == {a, b} for a, b, _ in pairs):
            pairs.append(
                (
                    primary,
                    secondary,
                    "face_each_other"
                    if focus_mode == "viewing"
                    else "access_faces_other",
                )
            )
    return pairs


def _pair_alignment_signals(
    *,
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pairs = _viewing_pairs_from_relation_plan(relation_plan)
    if not pairs:
        return {
            "pair_axis_ok": True,
            "primary_front_ok": True,
            "pair_alignment_score": 1.0,
            "primary_front_score": 1.0,
            "required": False,
        }
    rows = _cluster_rows_from_verify(verify)
    primary_cluster_id = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    pair_scores: list[float] = []
    primary_scores: list[float] = []
    for left, right, relation in pairs:
        left_row = rows.get(left)
        right_row = rows.get(right)
        if left_row is None or right_row is None:
            continue
        left_center = _cluster_center_tuple(left_row)
        right_center = _cluster_center_tuple(right_row)
        left_front = _vector_tuple(left_row.get("front_world"))
        right_front = _vector_tuple(right_row.get("front_world"))
        if left_center is None or right_center is None:
            continue
        forward = _normalized_direction(left_center, right_center)
        backward = None if forward is None else (-forward[0], -forward[1])
        left_score = max(-1.0, min(1.0, _dot(left_front, forward)))
        right_score = max(-1.0, min(1.0, _dot(right_front, backward)))
        if relation == "face_each_other":
            pair_scores.append((left_score + right_score) / 2.0)
        else:
            pair_scores.append(left_score)
        if primary_cluster_id:
            if left == primary_cluster_id:
                primary_scores.append(left_score)
            elif right == primary_cluster_id:
                primary_scores.append(right_score)
    if not pair_scores:
        return {
            "pair_axis_ok": True,
            "primary_front_ok": True,
            "pair_alignment_score": 1.0,
            "primary_front_score": 1.0,
            "required": bool(pairs),
        }
    pair_alignment_score = max(0.0, min(1.0, (min(pair_scores) + 1.0) / 2.0))
    if primary_scores:
        primary_front_score = max(
            0.0, min(1.0, (sum(primary_scores) / len(primary_scores) + 1.0) / 2.0)
        )
    else:
        primary_front_score = pair_alignment_score
    return {
        "pair_axis_ok": pair_alignment_score >= 0.72,
        "primary_front_ok": primary_front_score >= 0.70,
        "pair_alignment_score": pair_alignment_score,
        "primary_front_score": primary_front_score,
        "required": bool(pairs),
    }


def _solution_diagnostics(
    *,
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
    chosen: Mapping[str, Candidate],
) -> dict[str, bool]:
    quality = _quality_dict(verify)
    circulation_penalty = _as_int(quality.get("circulation_penalty_mm"))
    critical_penalty = _as_int(quality.get("critical_orientation_penalty_mm"))
    center_strength = _center_openness_strength(relation_plan)
    center_is_strong = center_strength in {"high", "very_high"}
    center_ok_threshold = 120 if center_is_strong else 260
    pair_signals = _pair_alignment_signals(
        verify=verify,
        relation_plan=relation_plan,
    )
    workflow_ok = critical_penalty <= 220
    if pair_signals.get("required"):
        workflow_ok = workflow_ok and bool(pair_signals.get("pair_axis_ok"))
    return {
        "circulation_clear": circulation_penalty <= 80,
        "entry_preserved": circulation_penalty <= 120,
        "center_openness_preserved": circulation_penalty <= center_ok_threshold,
        "dominant_anchor_correct": _dominant_anchor_correct(
            relation_plan=relation_plan,
            chosen=chosen,
        ),
        "primary_pair_axis_aligned": bool(pair_signals.get("pair_axis_ok")),
        "anchor_front_matches_pair_direction": bool(
            pair_signals.get("primary_front_ok")
        ),
        "workflow_preserved": workflow_ok,
    }


def _quality_gate_reasons_from_verify(
    verify: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(verify, Mapping):
        return []
    quality_gate = verify.get("quality_gate")
    if not isinstance(quality_gate, Mapping):
        return []
    return [
        str(reason or "")
        for reason in (quality_gate.get("reasons") or [])
        if str(reason or "").strip()
    ]


def _quality_gate_reasons_from_solution(solution: Mapping[str, Any]) -> list[str]:
    verify = solution.get("_verify")
    if isinstance(verify, Mapping):
        reasons = _quality_gate_reasons_from_verify(verify)
        if reasons:
            return reasons
    summary = solution.get("verify_summary")
    if not isinstance(summary, Mapping):
        return []
    return [
        str(reason or "")
        for reason in (summary.get("quality_gate_reasons") or [])
        if str(reason or "").strip()
    ]


def _selected_variant_looks_like_family(
    selected_variant: Mapping[str, Any],
    family: str,
) -> bool:
    family_token = _normalized_policy_token(family)
    variant_family = _normalized_policy_token(selected_variant.get("variant_family"))
    variant_id = _normalized_policy_token(selected_variant.get("variant_id"))
    return variant_family == family_token or family_token in variant_id


def _protected_topology_violations_from_verify(
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(verify, Mapping):
        return []
    quality = _quality_dict(verify)
    circulation_debug = quality.get("circulation_debug")
    if not isinstance(circulation_debug, list):
        return []

    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    preserve_entry = True
    preserve_corridor = True
    if isinstance(topology_policy, Mapping):
        preserve_entry = bool(topology_policy.get("preserve_entry_landing", True))
        preserve_corridor = bool(topology_policy.get("preserve_primary_corridor", True))

    violations: list[str] = []
    for item in circulation_debug:
        if not isinstance(item, Mapping):
            continue
        penalty = _as_int(item.get("penalty_mm"))
        if penalty <= 0:
            continue
        kind = str(item.get("kind") or "").strip()
        if kind == "entry_buffer" and preserve_entry:
            violations.append("entry_landing_zone_blocked")
        elif kind == "main_path" and preserve_corridor:
            violations.append("primary_circulation_corridor_blocked")
        elif kind == "center_lane" and _center_openness_is_very_high(relation_plan):
            violations.append("center_openness_region_blocked")
    return sorted(set(violations))


def _solution_publishability_violations(
    solution: Mapping[str, Any],
    *,
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    violations: list[str] = []
    if not bool(solution.get("hard_valid")):
        violations.append("not_hard_valid")

    gate_reasons = set(_quality_gate_reasons_from_solution(solution))
    violations.extend(sorted(gate_reasons & PUBLISHABLE_BLOCKING_QUALITY_REASONS))

    diagnostics_raw = solution.get("diagnostics")
    diagnostics = diagnostics_raw if isinstance(diagnostics_raw, Mapping) else {}
    for key in ("circulation_clear", "entry_preserved", "workflow_preserved"):
        if diagnostics.get(key) is False:
            violations.append(f"{key}_false")
    if diagnostics.get("dominant_anchor_correct") is False:
        violations.append("dominant_anchor_correct_false")
    if diagnostics.get("primary_pair_axis_aligned") is False:
        violations.append("primary_pair_axis_aligned_false")
    if diagnostics.get("anchor_front_matches_pair_direction") is False:
        violations.append("anchor_front_matches_pair_direction_false")
    if (
        _center_openness_is_strong(relation_plan)
        and diagnostics.get("center_openness_preserved") is False
    ):
        violations.append("center_openness_preserved_false")

    verify = solution.get("_verify")
    if isinstance(verify, Mapping):
        violations.extend(
            _protected_topology_violations_from_verify(verify, relation_plan)
        )

    selected_variants = [
        item
        for item in (solution.get("selected_variants") or [])
        if isinstance(item, Mapping)
    ]
    for item in selected_variants:
        cluster_id = str(item.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        if not _selected_variant_looks_like_family(item, "fallback_generic"):
            continue
        if _concept_allows_fallback_generic(relation_plan, cluster_id):
            continue
        if _cluster_is_core_or_protected(relation_plan, cluster_id):
            violations.append("fallback_generic_core_variant")

    return sorted(set(violations))


def _is_publishable_solution(
    solution: Mapping[str, Any],
    *,
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    return not _solution_publishability_violations(
        solution,
        relation_plan=relation_plan,
    )


def _global_quality_scores(
    *,
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
    chosen: Mapping[str, Candidate],
) -> dict[str, float]:
    quality = _quality_dict(verify)
    summary = _summary_dict(verify)

    circulation = _as_int(quality.get("circulation_penalty_mm"))
    tight_gap = _as_int(quality.get("tight_gap_penalty_mm"))
    affinity = _as_int(quality.get("affinity_penalty_mm"))
    relation = _as_int(quality.get("relation_penalty_mm"))
    spread = _as_int(quality.get("spread_penalty_mm"))
    orientation = _as_int(quality.get("orientation_penalty_mm"))
    critical = _as_int(quality.get("critical_orientation_penalty_mm"))
    focal = _as_int(quality.get("focal_pair_penalty_mm"))
    macro = _as_int(quality.get("macro_penalty_mm"))
    micro = _as_int(quality.get("micro_penalty_mm"))

    density = _as_float(quality.get("density_ratio"), 0.28)
    target_density = _style_target_density_ratio(relation_plan)
    density_score = _clamp_score(1.0 - (abs(density - target_density) / 0.32))
    min_gap = summary.get("min_cluster_gap_mm")
    gap_score = 0.72 if min_gap is None else _clamp_score(_as_float(min_gap) / 450.0)

    diagnostics = _solution_diagnostics(
        verify=verify,
        relation_plan=relation_plan,
        chosen=chosen,
    )
    pair_signals = _pair_alignment_signals(
        verify=verify,
        relation_plan=relation_plan,
    )
    dominant_anchor_score = 1.0 if diagnostics["dominant_anchor_correct"] else 0.05
    workflow_score = 1.0 if diagnostics["workflow_preserved"] else 0.05
    center_score = 1.0 if diagnostics["center_openness_preserved"] else 0.25
    pair_alignment_score = _clamp_score(
        float(pair_signals.get("pair_alignment_score") or 0.0)
    )
    primary_front_score = _clamp_score(
        float(pair_signals.get("primary_front_score") or 0.0)
    )
    concept_family = _concept_family_from_relation_plan(relation_plan)
    compatibility_values = [
        _variant_family_compatibility_score(
            concept_family=concept_family,
            role_kind=_cluster_role_kind(relation_plan, cluster_id),
            variant_family=candidate.variant_family,
        )
        for cluster_id, candidate in chosen.items()
    ]
    compatibility_score = _clamp_score(
        0.5 + (sum(compatibility_values) / max(1, len(compatibility_values))) / 2.4
    )
    gate_reasons = set(_quality_gate_reasons_from_verify(verify))
    primary_pair_ok = ("focal_pair_penalty_too_high" not in gate_reasons) and bool(
        pair_signals.get("pair_axis_ok")
    )
    critical_items_ok = "critical_item_penalty_too_high" not in gate_reasons

    functionality = (
        0.30 * _penalty_score(circulation, 420.0)
        + 0.18 * _penalty_score(tight_gap, 520.0)
        + 0.20 * _penalty_score(critical, 220.0)
        + 0.12 * _penalty_score(micro, 520.0)
        + 0.10 * gap_score
        + 0.10 * workflow_score
    )
    naturalness = (
        0.22 * _penalty_score(affinity, 520.0)
        + 0.18 * _penalty_score(spread, 680.0)
        + 0.16 * _penalty_score(orientation, 760.0)
        + 0.10 * _penalty_score(macro, 900.0)
        + 0.11 * density_score
        + 0.11 * dominant_anchor_score
        + 0.06 * pair_alignment_score
        + 0.06 * primary_front_score
    )
    semantic = (
        0.13 * _penalty_score(relation, 520.0)
        + 0.10 * _penalty_score(affinity, 520.0)
        + 0.14 * _penalty_score(critical, 220.0)
        + 0.14 * _penalty_score(focal, 220.0)
        + 0.10 * dominant_anchor_score
        + 0.08 * compatibility_score
        + 0.08 * workflow_score
        + 0.05 * center_score
        + 0.10 * pair_alignment_score
        + 0.06 * primary_front_score
        + 0.02 * (1.0 if primary_pair_ok else 0.0)
        + 0.00 * (1.0 if critical_items_ok else 0.0)
    )
    spatial = (
        0.18 * _penalty_score(tight_gap, 520.0)
        + 0.16 * _penalty_score(spread, 680.0)
        + 0.16 * _penalty_score(circulation, 500.0)
        + 0.14 * density_score
        + 0.14 * gap_score
        + 0.14 * center_score
        + 0.08 * pair_alignment_score
    )

    if not (isinstance(verify, Mapping) and bool(verify.get("hard_valid"))):
        functionality *= 0.35
        semantic *= 0.45
        spatial *= 0.45

    if pair_signals.get("required") and not pair_signals.get("pair_axis_ok"):
        semantic *= 0.52
        naturalness *= 0.72
    if pair_signals.get("required") and not pair_signals.get("primary_front_ok"):
        semantic *= 0.78

    quality_weights = _style_quality_weights(relation_plan)
    total = (
        quality_weights["functionality"] * functionality
        + quality_weights["naturalness"] * naturalness
        + quality_weights["semantic"] * semantic
        + quality_weights["spatial"] * spatial
    )
    return {
        "functionality_score": round(_clamp_score(functionality), 3),
        "naturalness_score": round(_clamp_score(naturalness), 3),
        "semantic_coherence_score": round(_clamp_score(semantic), 3),
        "spatial_quality_score": round(_clamp_score(spatial), 3),
        "total_score": round(_clamp_score(total), 3),
    }


def _style_target_density_ratio(relation_plan: Mapping[str, Any] | None) -> float:
    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    if not isinstance(topology_policy, Mapping):
        return 0.28
    text = str(topology_policy.get("style_density_target") or "").lower()
    if "low" in text:
        return 0.24
    if "medium_high" in text or "high" in text:
        return 0.34
    if "medium" in text:
        return 0.31
    return 0.28


def _style_quality_weights(relation_plan: Mapping[str, Any] | None) -> dict[str, float]:
    weights = dict(QUALITY_WEIGHTS)
    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    if not isinstance(topology_policy, Mapping):
        return weights
    reserve_center = str(topology_policy.get("reserve_center_degree") or "")
    if reserve_center in {"high", "very_high"}:
        weights["naturalness"] += 0.03
        weights["spatial"] += 0.03
        weights["functionality"] -= 0.03
        weights["semantic"] -= 0.03
    visual_balance = str(topology_policy.get("style_visual_balance") or "")
    if visual_balance in {"formal", "calm"}:
        weights["semantic"] += 0.02
        weights["naturalness"] += 0.02
        weights["functionality"] -= 0.02
        weights["spatial"] -= 0.02
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _normalized_text_set(values: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(values, list):
        return out
    for item in values:
        text = str(item or "").strip().lower()
        if text:
            out.add(text)
    return out


def _relation_priority_weight(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text == "critical":
        return 1.4
    if text == "high":
        return 1.0
    if text == "medium":
        return 0.7
    if text == "low":
        return 0.4
    return 0.8


def _cluster_relation_weight(relation: str) -> float:
    lowered = relation.strip().lower()
    if lowered == "near":
        return 4.0
    if lowered in {"adjacent", "next_to"}:
        return 3.4
    if lowered in {"aligned", "parallel"}:
        return 3.0
    return 2.6


def _cluster_directional_weight(relation: str) -> float:
    lowered = relation.strip().lower()
    if lowered == "face_each_other":
        return 8.0
    if lowered in {"face_same_direction", "axis_parallel"}:
        return 5.5
    if lowered in {"perpendicular", "orthogonal"}:
        return 2.4
    return 4.0


def _extract_layout_primary_cluster_id(
    relation_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    layout_intent = relation_plan.get("layout_intent_profile")
    if not isinstance(layout_intent, dict):
        return None
    cluster_id = str(layout_intent.get("primary_cluster_id") or "").strip()
    return cluster_id or None


def _extract_layout_secondary_cluster_id(
    relation_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    layout_intent = relation_plan.get("layout_intent_profile")
    if not isinstance(layout_intent, dict):
        return None
    cluster_id = str(layout_intent.get("secondary_cluster_id") or "").strip()
    return cluster_id or None


def _normalized_cluster_pair(a: str, b: str) -> tuple[str, str] | None:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right or left == right:
        return None
    return tuple(sorted((left, right)))


def _viewing_contract_pairs(
    relation_plan: dict[str, Any] | None,
) -> set[tuple[str, str]]:
    if not isinstance(relation_plan, dict):
        return set()

    pairs: set[tuple[str, str]] = set()
    primary_cluster_id = _extract_layout_primary_cluster_id(relation_plan)
    secondary_cluster_id = _extract_layout_secondary_cluster_id(relation_plan)
    layout_intent = relation_plan.get("layout_intent_profile") or {}
    focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
    if focus_mode in {"viewing", "mixed"}:
        pair = _normalized_cluster_pair(
            primary_cluster_id or "",
            secondary_cluster_id or "",
        )
        if pair is not None:
            pairs.add(pair)

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"face_each_other", "access_faces_other"}:
            continue
        pair = _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        if pair is not None:
            pairs.add(pair)

    return pairs


def _viewing_contract_cluster_ids(
    relation_plan: dict[str, Any] | None,
) -> set[str]:
    cluster_ids: set[str] = set()
    for left, right in _viewing_contract_pairs(relation_plan):
        cluster_ids.add(left)
        cluster_ids.add(right)
    return cluster_ids


def _cluster_face_target_id(
    relation_plan: dict[str, Any] | None,
    cluster_id: str,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    target: str | None = None
    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        intents = _normalized_text_set(row.get("intents"))
        if "face_cluster" not in intents:
            continue
        target = str(row.get("target_cluster_id") or "").strip() or None
        if target is not None:
            return target
    return None


def _build_cluster_macro_profiles(
    relation_plan: dict[str, Any] | None,
    *,
    cluster_ids: Sequence[str],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, float]]]:
    clean_cluster_ids = [str(cluster_id or "").strip() for cluster_id in cluster_ids]
    clean_cluster_ids = [cluster_id for cluster_id in clean_cluster_ids if cluster_id]
    profiles = {
        cluster_id: {"prefer": set(), "avoid": set(), "intents": set()}
        for cluster_id in clean_cluster_ids
    }
    adjacency: dict[str, dict[str, float]] = {
        cluster_id: {} for cluster_id in clean_cluster_ids
    }
    if not isinstance(relation_plan, dict):
        return profiles, adjacency

    def ensure(cluster_id: str) -> dict[str, set[str]] | None:
        if cluster_id not in profiles:
            return None
        return profiles[cluster_id]

    def add_edge(a: str, b: str, weight: float) -> None:
        if a == b or a not in adjacency or b not in adjacency or weight <= 0.0:
            return
        adjacency[a][b] = adjacency[a].get(b, 0.0) + float(weight)
        adjacency[b][a] = adjacency[b].get(a, 0.0) + float(weight)

    for row in relation_plan.get("cluster_affinities") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        slot = ensure(cluster_id)
        if slot is None:
            continue
        slot["prefer"].update(_normalized_text_set(row.get("prefer")))
        slot["avoid"].update(_normalized_text_set(row.get("avoid")))

    _apply_macro_concept_profiles(
        relation_plan,
        profiles=profiles,
    )

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        slot = ensure(cluster_id)
        if slot is None:
            continue
        intents = _normalized_text_set(row.get("intents"))
        slot["intents"].update(intents)
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        if target_cluster_id and "face_cluster" in intents:
            add_edge(
                cluster_id,
                target_cluster_id,
                5.5 * _relation_priority_weight(row.get("priority")),
            )

    for row in relation_plan.get("cluster_relations") or []:
        if not isinstance(row, dict):
            continue
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if not a or not b:
            continue
        add_edge(
            a,
            b,
            _cluster_relation_weight(str(row.get("relation") or ""))
            * _relation_priority_weight(row.get("priority")),
        )

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if not a or not b:
            continue
        add_edge(
            a,
            b,
            _cluster_directional_weight(str(row.get("relation") or ""))
            * _relation_priority_weight(row.get("priority")),
        )
    return profiles, adjacency


def _apply_macro_concept_profiles(
    relation_plan: dict[str, Any],
    *,
    profiles: dict[str, dict[str, set[str]]],
) -> None:
    macro_concept = relation_plan.get("macro_concept")
    if not isinstance(macro_concept, dict):
        return

    topology_policy = macro_concept.get("topology_policy")
    reserve_center = False
    if isinstance(topology_policy, dict):
        reserve_center = str(
            topology_policy.get("reserve_center_degree") or ""
        ).strip() in {"high", "very_high"}

    zone_plan = macro_concept.get("cluster_zone_plan")
    if not isinstance(zone_plan, list):
        return

    for row in zone_plan:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        profile = profiles.get(cluster_id)
        if profile is None:
            continue

        zone_assignment = str(row.get("zone_assignment") or "").strip()
        wall_claim = str(row.get("wall_claim") or "").strip()
        center_usage = str(row.get("center_usage") or "").strip()
        entry_relation = str(row.get("entry_relation") or "").strip()
        daylight_relation = str(row.get("daylight_relation") or "").strip()

        if wall_claim in {"strong", "medium"} or any(
            token in zone_assignment for token in ("wall", "edge", "storage")
        ):
            profile["prefer"].update({"wall", "recess_or_edge"})
            profile["intents"].add("back_to_wall")
        if wall_claim == "strong":
            profile["prefer"].add("long_wall")
            profile["intents"].add("axis_parallel_wall")
        if "daylight" in zone_assignment or daylight_relation in {
            "claim_daylight",
            "daylight_preferred",
        }:
            profile["prefer"].add("window_side")
        if daylight_relation == "avoid_window_blocking":
            profile["avoid"].add("window_blocking")
        if (
            "private" in zone_assignment
            or entry_relation == "avoid_direct_entry_conflict"
        ):
            profile["prefer"].add("far_from_entry")
            profile["avoid"].add("entry_blocking")
        if reserve_center or center_usage in {"none", "open_reserved"}:
            profile["avoid"].add("center")
        elif center_usage in {"partial", "primary"}:
            profile["prefer"].add("center")


def _cluster_prefers_wall(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    intents = set(profile.get("intents") or set())
    return bool((prefer & _CLUSTER_WALL_PREFER_TAGS) or {"back_to_wall"} & intents)


def _cluster_prefers_center(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    return bool(prefer & _CLUSTER_CENTER_PREFER_TAGS)


def _cluster_avoids_center(profile: Mapping[str, set[str]]) -> bool:
    avoid = set(profile.get("avoid") or set())
    return bool(avoid & _CLUSTER_CENTER_AVOID_TAGS)


def _cluster_avoids_window(profile: Mapping[str, set[str]]) -> bool:
    avoid = set(profile.get("avoid") or set())
    return bool(avoid & _CLUSTER_WINDOW_AVOID_TAGS)


def _cluster_avoids_entry(profile: Mapping[str, set[str]]) -> bool:
    avoid = set(profile.get("avoid") or set())
    prefer = set(profile.get("prefer") or set())
    return bool(avoid & _CLUSTER_ENTRY_AVOID_TAGS) or ("far_from_entry" in prefer)


def _cluster_prefers_window(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    intents = set(profile.get("intents") or set())
    return bool(prefer & _CLUSTER_WINDOW_PREFER_TAGS) or ("face_window" in intents)


def _cluster_prefers_entry(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    intents = set(profile.get("intents") or set())
    return bool(prefer & _CLUSTER_ENTRY_PREFER_TAGS) or ("face_entry" in intents)


def _anchor_kind_contains(anchor_kind: str, tokens: Sequence[str]) -> bool:
    lowered = anchor_kind.strip().lower()
    return any(token in lowered for token in tokens)


def _anchor_kind_is_wall_pinned(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _WALL_ANCHOR_TOKENS)


def _anchor_kind_is_center(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _CENTER_ANCHOR_TOKENS)


def _anchor_kind_is_window(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _WINDOW_ANCHOR_TOKENS)


def _anchor_kind_is_entry(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _ENTRY_ANCHOR_TOKENS)


def _cluster_root_score(
    *,
    cluster_id: str,
    profile: Mapping[str, set[str]],
    adjacency: Mapping[str, Mapping[str, float]],
    candidate_count: int,
    original_index: int,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> float:
    prefer = set(profile.get("prefer") or set())
    avoid = set(profile.get("avoid") or set())
    intents = set(profile.get("intents") or set())
    score = 0.0
    if cluster_id == primary_cluster_id:
        score += 120.0
    if cluster_id == secondary_cluster_id:
        score += 26.0
    if "back_to_wall" in intents:
        score += 22.0
    if "wall" in prefer:
        score += 16.0
    if "long_wall" in prefer:
        score += 12.0
    if "short_wall" in prefer:
        score += 6.0
    if "recess_or_edge" in prefer:
        score += 6.0
    if "access_to_open_space" in intents:
        score += 5.0
    if "inward_to_room" in intents:
        score += 4.0
    if "far_from_entry" in prefer:
        score += 4.0
    score += min(18.0, 3.0 * float(len(avoid)))
    score += min(34.0, float(sum(adjacency.get(cluster_id, {}).values())))
    score += min(14.0, 32.0 / max(2.0, float(candidate_count)))
    score += max(0.0, 3.0 - (0.18 * float(original_index)))
    return score


def _select_root_cluster_id(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    relation_plan: dict[str, Any] | None,
    cluster_order: Sequence[str] | None = None,
) -> str | None:
    ordered_ids = _ordered_cluster_ids(
        candidates_by_cluster, cluster_order=cluster_order
    )
    if not ordered_ids:
        return None
    primary_cluster_id = _extract_layout_primary_cluster_id(relation_plan)
    if primary_cluster_id in candidates_by_cluster:
        return primary_cluster_id

    secondary_cluster_id = _extract_layout_secondary_cluster_id(relation_plan)
    profiles, adjacency = _build_cluster_macro_profiles(
        relation_plan,
        cluster_ids=ordered_ids,
    )
    original_index = {cluster_id: idx for idx, cluster_id in enumerate(ordered_ids)}
    return max(
        ordered_ids,
        key=lambda cluster_id: (
            _cluster_root_score(
                cluster_id=cluster_id,
                profile=profiles.get(cluster_id, {}),
                adjacency=adjacency,
                candidate_count=len(candidates_by_cluster.get(cluster_id, ())),
                original_index=original_index.get(cluster_id, 0),
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
            ),
            -original_index.get(cluster_id, 0),
        ),
    )


def _root_cluster_first_order(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    relation_plan: dict[str, Any] | None,
    cluster_order: Sequence[str] | None = None,
) -> list[str]:
    ordered_ids = _ordered_cluster_ids(
        candidates_by_cluster, cluster_order=cluster_order
    )
    if not ordered_ids:
        return []
    root_cluster_id = _select_root_cluster_id(
        candidates_by_cluster,
        relation_plan=relation_plan,
        cluster_order=cluster_order,
    )
    if root_cluster_id is None:
        return ordered_ids

    profiles, adjacency = _build_cluster_macro_profiles(
        relation_plan,
        cluster_ids=ordered_ids,
    )
    primary_cluster_id = _extract_layout_primary_cluster_id(relation_plan)
    secondary_cluster_id = _extract_layout_secondary_cluster_id(relation_plan)
    original_index = {cluster_id: idx for idx, cluster_id in enumerate(ordered_ids)}
    root_scores = {
        cluster_id: _cluster_root_score(
            cluster_id=cluster_id,
            profile=profiles.get(cluster_id, {}),
            adjacency=adjacency,
            candidate_count=len(candidates_by_cluster.get(cluster_id, ())),
            original_index=original_index.get(cluster_id, 0),
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        for cluster_id in ordered_ids
    }

    placed = [root_cluster_id]
    placed_set = {root_cluster_id}
    remaining = {
        cluster_id for cluster_id in ordered_ids if cluster_id != root_cluster_id
    }
    while remaining:
        connected: list[tuple[float, float, int, str]] = []
        for cluster_id in remaining:
            edge_strength = max(
                (
                    float(adjacency.get(cluster_id, {}).get(parent_id, 0.0))
                    for parent_id in placed_set
                ),
                default=0.0,
            )
            if edge_strength <= 0.0:
                continue
            connected.append(
                (
                    edge_strength,
                    root_scores.get(cluster_id, 0.0),
                    -original_index.get(cluster_id, 0),
                    cluster_id,
                )
            )
        if connected:
            next_cluster_id = max(connected)[3]
        else:
            next_cluster_id = max(
                remaining,
                key=lambda cluster_id: (
                    root_scores.get(cluster_id, 0.0),
                    -original_index.get(cluster_id, 0),
                ),
            )
        placed.append(next_cluster_id)
        placed_set.add(next_cluster_id)
        remaining.remove(next_cluster_id)
    return placed


def _candidate_macro_bonus(
    candidate: Candidate,
    *,
    profile: Mapping[str, set[str]],
    is_root_cluster: bool,
) -> int:
    scale = 2 if is_root_cluster else 1
    bonus = 0
    anchor_kind = str(candidate.anchor_kind or "")
    if _cluster_prefers_wall(profile):
        wall_weight = 1600 * scale
        if _anchor_kind_is_wall_pinned(anchor_kind):
            bonus += wall_weight
            if "long_wall" in set(profile.get("prefer") or set()):
                bonus += 400 * scale
        else:
            bonus -= wall_weight // 2
    if _cluster_avoids_center(profile) and _anchor_kind_is_center(anchor_kind):
        bonus -= 800 * scale
    if _cluster_prefers_center(profile) and _anchor_kind_is_center(anchor_kind):
        bonus += 500 * scale
    if _cluster_avoids_window(profile) and _anchor_kind_is_window(anchor_kind):
        bonus -= 700 * scale
    if _cluster_prefers_window(profile) and _anchor_kind_is_window(anchor_kind):
        bonus += 500 * scale
    if _cluster_avoids_entry(profile) and _anchor_kind_is_entry(anchor_kind):
        bonus -= 750 * scale
    if _cluster_prefers_entry(profile) and _anchor_kind_is_entry(anchor_kind):
        bonus += 450 * scale
    return bonus


def _candidate_variant_family_bonus(
    candidate: Candidate,
    *,
    relation_plan: Mapping[str, Any] | None,
) -> int:
    compatibility = _variant_family_compatibility_score(
        concept_family=_concept_family_from_relation_plan(relation_plan),
        role_kind=_cluster_role_kind(relation_plan, candidate.cluster_id),
        variant_family=candidate.variant_family,
    )
    scale = (
        2400
        if _cluster_is_core_or_protected(relation_plan, candidate.cluster_id)
        else 1300
    )
    return int(round(compatibility * scale))


def _has_orientation_sensitive_wall_contract(profile: Mapping[str, set[str]]) -> bool:
    intents = set(profile.get("intents") or set())
    return bool(
        intents
        & {"back_to_wall", "face_cluster", "access_to_open_space", "inward_to_room"}
    )


def _has_hard_root_wall_contract(
    *,
    relation_plan: dict[str, Any] | None,
    root_cluster_id: str | None,
    profile: Mapping[str, set[str]],
) -> bool:
    if not root_cluster_id or not _cluster_prefers_wall(profile):
        return False
    intents = set(profile.get("intents") or set())
    target_cluster_id = _cluster_face_target_id(relation_plan, root_cluster_id)
    if "back_to_wall" not in intents or not target_cluster_id:
        return False
    if "face_cluster" not in intents:
        return False
    protected_cluster_ids = _viewing_contract_cluster_ids(relation_plan)
    return (
        root_cluster_id in protected_cluster_ids
        or target_cluster_id in protected_cluster_ids
        or target_cluster_id == _extract_layout_secondary_cluster_id(relation_plan)
    )


def _orientation_focus_candidates(
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    if not candidates:
        return []
    best_critical = min(
        int(candidate.critical_orientation_penalty_mm) for candidate in candidates
    )
    best_focal = min(
        int(candidate.focal_orientation_penalty_mm) for candidate in candidates
    )
    best_orientation = min(
        int(candidate.orientation_penalty_mm) for candidate in candidates
    )
    focused = [
        candidate
        for candidate in candidates
        if candidate.acceptable_valid
        or (
            int(candidate.critical_orientation_penalty_mm) <= best_critical + 120
            and int(candidate.focal_orientation_penalty_mm) <= best_focal + 160
            and int(candidate.orientation_penalty_mm) <= best_orientation + 240
        )
    ]
    return focused or list(candidates)


def _hard_contract_focus_candidates(
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    if not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.acceptable_valid else 1,
            int(candidate.critical_orientation_penalty_mm),
            int(candidate.focal_orientation_penalty_mm),
            int(candidate.orientation_penalty_mm),
            -int(candidate.rough_score),
            candidate.variant_id,
            candidate.rot,
            candidate.x,
            candidate.y,
        ),
    )
    best_critical = int(ranked[0].critical_orientation_penalty_mm)
    best_focal = int(ranked[0].focal_orientation_penalty_mm)
    best_orientation = int(ranked[0].orientation_penalty_mm)
    acceptable = [
        candidate
        for candidate in ranked
        if candidate.acceptable_valid
        and int(candidate.critical_orientation_penalty_mm) <= best_critical + 40
        and int(candidate.focal_orientation_penalty_mm) <= best_focal + 80
        and int(candidate.orientation_penalty_mm) <= best_orientation + 120
    ]
    if acceptable:
        return acceptable[:12]
    focused = [
        candidate
        for candidate in ranked
        if int(candidate.critical_orientation_penalty_mm) <= best_critical + 20
        and int(candidate.focal_orientation_penalty_mm) <= best_focal + 40
        and int(candidate.orientation_penalty_mm) <= best_orientation + 60
    ]
    return (focused or ranked)[:8]


def _wall_pin_focus_next_action(
    *,
    round_strategy: Mapping[str, Any] | None,
) -> str:
    if not isinstance(round_strategy, Mapping):
        return "none"
    if not bool(round_strategy.get("used_wall_pin_focus")):
        return "none"
    return "refine_with_focus"


def _prepare_round_candidates(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    relation_plan: dict[str, Any] | None,
    cluster_order: Sequence[str] | None = None,
    wall_pin_focus: bool,
) -> tuple[
    dict[str, list[Candidate]],
    list[str],
    dict[tuple[str, str, int, int, int], int],
    dict[str, Any],
]:
    ordered_cluster_ids = _root_cluster_first_order(
        candidates_by_cluster,
        relation_plan=relation_plan,
        cluster_order=cluster_order,
    )
    root_cluster_id = _select_root_cluster_id(
        candidates_by_cluster,
        relation_plan=relation_plan,
        cluster_order=ordered_cluster_ids,
    )
    profiles, _adjacency = _build_cluster_macro_profiles(
        relation_plan,
        cluster_ids=ordered_cluster_ids,
    )
    candidate_bonus_by_key: dict[tuple[str, str, int, int, int], int] = {}
    for cluster_id in ordered_cluster_ids:
        profile = profiles.get(cluster_id, {})
        is_root_cluster = cluster_id == root_cluster_id
        for candidate in candidates_by_cluster.get(cluster_id, ()):
            candidate_bonus_by_key[candidate.key] = _candidate_macro_bonus(
                candidate,
                profile=profile,
                is_root_cluster=is_root_cluster,
            ) + _candidate_variant_family_bonus(
                candidate,
                relation_plan=relation_plan,
            )

    prioritized: dict[str, list[Candidate]] = {}
    used_wall_pin_focus = False
    used_hard_wall_contract = False
    for cluster_id in ordered_cluster_ids:
        sorted_candidates = sorted(
            list(candidates_by_cluster.get(cluster_id, ())),
            key=lambda candidate: (
                -_candidate_effective_score(candidate, candidate_bonus_by_key),
                candidate.variant_id,
                candidate.rot,
                candidate.x,
                candidate.y,
            ),
        )
        profile = profiles.get(cluster_id, {})
        if (
            wall_pin_focus
            and cluster_id == root_cluster_id
            and _cluster_prefers_wall(profile)
        ):
            wall_candidates = [
                candidate
                for candidate in sorted_candidates
                if _anchor_kind_is_wall_pinned(candidate.anchor_kind)
            ]
            if len(wall_candidates) >= min(4, len(sorted_candidates)):
                if _has_hard_root_wall_contract(
                    relation_plan=relation_plan,
                    root_cluster_id=root_cluster_id,
                    profile=profile,
                ):
                    wall_candidates = _hard_contract_focus_candidates(wall_candidates)
                    used_hard_wall_contract = True
                elif _has_orientation_sensitive_wall_contract(profile):
                    wall_candidates = sorted(
                        _orientation_focus_candidates(wall_candidates),
                        key=lambda candidate: (
                            0 if candidate.acceptable_valid else 1,
                            int(candidate.critical_orientation_penalty_mm),
                            int(candidate.focal_orientation_penalty_mm),
                            int(candidate.orientation_penalty_mm),
                            -_candidate_effective_score(
                                candidate, candidate_bonus_by_key
                            ),
                            candidate.variant_id,
                            candidate.rot,
                            candidate.x,
                            candidate.y,
                        ),
                    )
                sorted_candidates = wall_candidates[: min(len(wall_candidates), 20)]
                used_wall_pin_focus = True
        prioritized[cluster_id] = sorted_candidates

    debug = {
        "root_cluster_id": root_cluster_id,
        "cluster_order": ordered_cluster_ids,
        "used_wall_pin_focus": used_wall_pin_focus,
        "used_hard_wall_contract": used_hard_wall_contract,
        "wall_pin_focus_requested": bool(wall_pin_focus),
    }
    return prioritized, ordered_cluster_ids, candidate_bonus_by_key, debug


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

        # Only explicit front-access requirements should become global object orientation intents.
        # A forge-facing declaration describes the asset's local front; it is not, by itself,
        # a requirement that the object front must point to room open space.
        for oid in sorted(access_ids):
            intents = tuple(sorted({"preserve_front_access", "front_to_open_space"}))
            sig = (cid, oid, intents)
            if sig in existing:
                continue
            base["object_orientations"].append(
                {
                    "cluster_id": cid,
                    "object_id": oid,
                    "intents": ["preserve_front_access", "front_to_open_space"],
                    "priority": "high",
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
) -> tuple[dict[str, list[Candidate]], list[str]]:
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
    cluster_order: list[str] = []
    for row in res.get("clusters") or []:
        cid = str(row.get("cluster_id") or "")
        if not cid:
            continue
        out[cid] = [_candidate_from_dict(x) for x in (row.get("candidates") or [])]
        cluster_order.append(cid)
    return out, cluster_order


def _merge_cluster_order(*orders: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for order in orders:
        for cluster_id in order:
            clean_cluster_id = str(cluster_id or "").strip()
            if not clean_cluster_id or clean_cluster_id in seen:
                continue
            merged.append(clean_cluster_id)
            seen.add(clean_cluster_id)
    return merged


def _ordered_cluster_ids(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    cluster_order: Sequence[str] | None = None,
) -> list[str]:
    preferred_order = _merge_cluster_order(cluster_order or (), candidates_by_cluster)
    return [
        cluster_id
        for cluster_id in preferred_order
        if cluster_id in candidates_by_cluster
    ]


def _relax_relation_plan_for_candidate_search(
    relation_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(relation_plan, dict):
        return relation_plan

    out = dict(relation_plan)
    protected_pairs = _viewing_contract_pairs(relation_plan)
    protected_cluster_ids = _viewing_contract_cluster_ids(relation_plan)
    out["cluster_relations"] = [
        dict(row)
        for row in relation_plan.get("cluster_relations") or []
        if isinstance(row, dict)
        and _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        in protected_pairs
    ]
    out["cluster_directional_relations"] = [
        dict(row)
        for row in relation_plan.get("cluster_directional_relations") or []
        if isinstance(row, dict)
        and _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        in protected_pairs
    ]

    cluster_orientations: list[dict[str, Any]] = []
    for entry in relation_plan.get("cluster_orientations") or []:
        if not isinstance(entry, dict):
            continue
        cluster_id = str(entry.get("cluster_id") or "").strip()
        if cluster_id in protected_cluster_ids:
            cluster_orientations.append(dict(entry))
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
        cluster_id = str(entry.get("cluster_id") or "").strip()
        target_cluster_id = str(entry.get("target_object_cluster_id") or "").strip()
        if (
            cluster_id in protected_cluster_ids
            and target_cluster_id in protected_cluster_ids
        ):
            object_orientations.append(dict(entry))
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
    cluster_order: Sequence[str] | None = None,
) -> dict[str, list[Candidate]]:
    relaxed_plan = _relax_relation_plan_for_candidate_search(relation_plan)
    extra_candidates, extra_cluster_order = _enumerate_initial_candidates(
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
    extra_candidates = _filter_candidates_by_concept_variant_policy(
        extra_candidates,
        relation_plan,
    )
    extra_candidates = _filter_protected_topology_candidates(
        extra_candidates,
        relation_plan,
    )

    merged: dict[str, list[Candidate]] = {}
    merged_order = _merge_cluster_order(
        cluster_order or tuple(current_candidates.keys()),
        extra_cluster_order,
        tuple(extra_candidates.keys()),
    )
    for cid in merged_order:
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


def _cap_candidate_search_space(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    max_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_CLUSTER,
    max_total: int = DEFAULT_MAX_TOTAL_BINARY_CANDIDATES,
) -> dict[str, list[Candidate]]:
    capped = {
        cluster_id: _dedupe_candidates(candidates)[: max(1, int(max_per_cluster))]
        for cluster_id, candidates in candidates_by_cluster.items()
    }
    while sum(len(candidates) for candidates in capped.values()) > max(
        1, int(max_total)
    ):
        largest_cluster_id = max(
            capped,
            key=lambda cluster_id: (
                len(capped[cluster_id]),
                cluster_id,
            ),
        )
        if len(capped[largest_cluster_id]) <= 1:
            break
        capped[largest_cluster_id] = capped[largest_cluster_id][:-1]
    return capped


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
    cluster_order: Sequence[str] | None = None,
    max_pair_bonus_abs: int = 12000,
) -> tuple[
    set[tuple[tuple[str, int], tuple[str, int]]],
    dict[tuple[tuple[str, int], tuple[str, int]], int],
]:
    incompatible: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    pair_bonus: dict[tuple[tuple[str, int], tuple[str, int]], int] = {}
    cluster_ids = _ordered_cluster_ids(
        candidates_by_cluster,
        cluster_order=cluster_order,
    )
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
                    pair_layout_score = _verify_search_score(pair_res)
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


def _iter_materialized_cluster_rows(clusters_outlines: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            return [dict(item) for item in clusters if isinstance(item, Mapping)]
        for cluster_id, payload in clusters_outlines.items():
            if not isinstance(payload, Mapping):
                continue
            row = dict(payload)
            row.setdefault("cluster_id", str(cluster_id))
            rows.append(row)
        return rows
    if isinstance(clusters_outlines, list):
        return [dict(item) for item in clusters_outlines if isinstance(item, Mapping)]
    return rows


def _parse_vec2_any(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        dx = value.get("dx")
        dy = value.get("dy")
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        dx, dy = value[0], value[1]
    else:
        return None
    try:
        x = float(dx)
        y = float(dy)
    except (TypeError, ValueError):
        return None
    norm = math.hypot(x, y)
    if norm <= 1e-6:
        return None
    return (x / norm, y / norm)


def _rotate_vec_ccw_90s(vec: tuple[float, float], rot: int) -> tuple[float, float]:
    x, y = float(vec[0]), float(vec[1])
    r = int(rot or 0) % 360
    if r == 0:
        return (x, y)
    if r == 90:
        return (-y, x)
    if r == 180:
        return (-x, -y)
    if r == 270:
        return (y, -x)
    return (x, y)


def _rotate_point_ccw_90s_local(x: float, y: float, rot: int) -> tuple[float, float]:
    r = int(rot or 0) % 360
    if r == 0:
        return (x, y)
    if r == 90:
        return (-y, x)
    if r == 180:
        return (-x, -y)
    if r == 270:
        return (y, -x)
    return (x, y)


def _cluster_local_bbox(
    payload: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    fp = payload.get("cluster_footprint")
    if isinstance(fp, Mapping):
        bbox = fp.get("local_bbox")
        if isinstance(bbox, Mapping):
            try:
                return (
                    float(bbox.get("min_x")),
                    float(bbox.get("min_y")),
                    float(bbox.get("max_x")),
                    float(bbox.get("max_y")),
                )
            except (TypeError, ValueError):
                pass
        rects = fp.get("rects")
        if isinstance(rects, Sequence) and not isinstance(rects, str):
            xs: list[float] = []
            ys: list[float] = []
            for rect in rects:
                if not isinstance(rect, Mapping):
                    continue
                try:
                    x = float(rect.get("x", 0))
                    y = float(rect.get("y", 0))
                    w = float(rect.get("w", 0))
                    h = float(rect.get("h", 0))
                except (TypeError, ValueError):
                    continue
                xs.extend([x, x + max(w, 0.0)])
                ys.extend([y, y + max(h, 0.0)])
            if xs and ys:
                return (min(xs), min(ys), max(xs), max(ys))
    outlines = payload.get("outline_polygons_ccw")
    if isinstance(outlines, Sequence) and not isinstance(outlines, str):
        xs: list[float] = []
        ys: list[float] = []
        for poly in outlines:
            if not isinstance(poly, Sequence) or isinstance(poly, str):
                continue
            for point in poly:
                if not isinstance(point, Mapping):
                    continue
                try:
                    xs.append(float(point.get("x")))
                    ys.append(float(point.get("y")))
                except (TypeError, ValueError):
                    continue
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _materialize_cluster_rows_from_selected_layout(
    *,
    clusters_outlines: Any,
    cluster_transforms: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads = {
        str(row.get("cluster_id") or ""): row
        for row in _iter_materialized_cluster_rows(clusters_outlines)
        if isinstance(row, Mapping) and str(row.get("cluster_id") or "").strip()
    }
    rows: list[dict[str, Any]] = []
    for transform in cluster_transforms:
        if not isinstance(transform, Mapping):
            continue
        cluster_id = str(transform.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        payload = payloads.get(cluster_id)
        if not isinstance(payload, Mapping):
            continue
        try:
            tx = float(transform.get("x", 0))
            ty = float(transform.get("y", 0))
            rot = int(transform.get("rot", 0)) % 360
        except (TypeError, ValueError):
            continue
        orientation_meta = payload.get("orientation_meta")
        orientation_meta = (
            orientation_meta if isinstance(orientation_meta, Mapping) else {}
        )
        front_local = _parse_vec2_any(
            orientation_meta.get("cluster_front_local")
            or orientation_meta.get("front_local")
        )
        front_world = None
        if front_local is not None:
            vec = _rotate_vec_ccw_90s(front_local, rot)
            front_world = {"dx": round(vec[0], 6), "dy": round(vec[1], 6)}
        bbox_local = _cluster_local_bbox(payload)
        bbox_world = None
        world_center = None
        if bbox_local is not None:
            min_x, min_y, max_x, max_y = bbox_local
            corners = [
                _rotate_point_ccw_90s_local(min_x, min_y, rot),
                _rotate_point_ccw_90s_local(max_x, min_y, rot),
                _rotate_point_ccw_90s_local(max_x, max_y, rot),
                _rotate_point_ccw_90s_local(min_x, max_y, rot),
            ]
            xs = [tx + pt[0] for pt in corners]
            ys = [ty + pt[1] for pt in corners]
            bbox_world = {
                "min_x": min(xs),
                "min_y": min(ys),
                "max_x": max(xs),
                "max_y": max(ys),
            }
            world_center = {
                "x": (bbox_world["min_x"] + bbox_world["max_x"]) / 2.0,
                "y": (bbox_world["min_y"] + bbox_world["max_y"]) / 2.0,
            }
        rows.append(
            {
                "cluster_id": cluster_id,
                "front_world": front_world,
                "bbox": bbox_world,
                "world_center": world_center,
            }
        )
    return rows


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
    result = verify(
        room_model=room_model,
        clusters_outlines=mat,
        cluster_transforms=cluster_transforms,
        grid_mm=grid_mm,
        mode="complete",
        relation_plan=relation_plan,
        return_debug=False,
    )
    if isinstance(result, Mapping) and not isinstance(
        result.get("materialized_layout"), Mapping
    ):
        result = dict(result)
        result["materialized_layout"] = {
            "clusters": _materialize_cluster_rows_from_selected_layout(
                clusters_outlines=mat,
                cluster_transforms=cluster_transforms,
            )
        }
    return result


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
            raw_variants = [
                v for v in (row.get("variants") or []) if isinstance(v, Mapping)
            ]
            semantic_families_available = any(
                _normalized_policy_token(v.get("family")) in SEMANTIC_VARIANT_FAMILIES
                for v in raw_variants
            )
            vids = [
                str(v.get("variant_id") or "")
                for v in raw_variants
                if _variant_family_allowed_for_cluster(
                    relation_plan=relation_plan,
                    cluster_id=cid,
                    variant_family=str(v.get("family") or ""),
                    semantic_families_available=semantic_families_available,
                )
            ]
            variant_ids_by_cluster[cid] = [v for v in vids if v]

    for cid in problematic_cluster_ids:
        current = chosen.get(cid)
        if current is None:
            continue
        existing = {c.key for c in new_map.get(cid, [])}
        proposals: list[Candidate] = []
        if cid in variant_ids_by_cluster:
            variant_ids = variant_ids_by_cluster[cid]
        else:
            variant_ids = [current.variant_id]
        if not variant_ids:
            continue
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
                        macro_penalty_mm=0,
                        micro_penalty_mm=0,
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
                        macro_penalty_mm=int(q.get("macro_penalty_mm") or 0),
                        micro_penalty_mm=int(q.get("micro_penalty_mm") or 0),
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
    cluster_order: Sequence[str],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    time_limit_s: float,
    num_workers: int,
    candidate_bonus_by_key: Mapping[tuple[str, str, int, int, int], int] | None = None,
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
        cluster_order=cluster_order,
    )

    model = cp_model.CpModel()
    xvars: dict[tuple[str, int], cp_model.IntVar] = {}
    ordered_pick_vars: list[cp_model.IntVar] = []
    for cid in _ordered_cluster_ids(candidates_by_cluster, cluster_order=cluster_order):
        cands = candidates_by_cluster[cid]
        vars_for_cluster = []
        for i, cand in enumerate(cands):
            var = model.NewBoolVar(f"pick[{cid},{i}]")
            xvars[(cid, i)] = var
            vars_for_cluster.append(var)
        model.AddExactlyOne(vars_for_cluster)
        ordered_pick_vars.extend(vars_for_cluster)

    for a_key, b_key in sorted(incompatible):
        model.Add(xvars[a_key] + xvars[b_key] <= 1)

    objective_terms: list[Any] = []
    for cid in _ordered_cluster_ids(candidates_by_cluster, cluster_order=cluster_order):
        cands = candidates_by_cluster[cid]
        for i, cand in enumerate(cands):
            objective_terms.append(
                _candidate_effective_score(cand, candidate_bonus_by_key)
                * xvars[(cid, i)]
            )

    for (a_key, b_key), bonus in pair_bonus.items():
        za = xvars[a_key]
        zb = xvars[b_key]
        z = model.NewBoolVar(f"pair[{a_key[0]},{a_key[1]}|{b_key[0]},{b_key[1]}]")
        model.Add(z <= za)
        model.Add(z <= zb)
        model.Add(z >= za + zb - 1)
        objective_terms.append(int(bonus) * z)

    model.Maximize(sum(objective_terms))

    add_decision_strategy = getattr(model, "AddDecisionStrategy", None)
    choose_first = getattr(cp_model, "CHOOSE_FIRST", None)
    select_max_value = getattr(cp_model, "SELECT_MAX_VALUE", None)
    if (
        callable(add_decision_strategy)
        and choose_first is not None
        and select_max_value is not None
        and ordered_pick_vars
    ):
        add_decision_strategy(ordered_pick_vars, choose_first, select_max_value)

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


def _solve_solution_pool_one_round(
    *,
    tools: Any,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    cluster_order: Sequence[str],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    time_limit_s: float,
    num_workers: int,
    candidate_bonus_by_key: Mapping[tuple[str, str, int, int, int], int] | None = None,
    max_solutions: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        cluster_order=cluster_order,
    )

    model = cp_model.CpModel()
    xvars: dict[tuple[str, int], cp_model.IntVar] = {}
    ordered_pick_vars: list[cp_model.IntVar] = []
    ordered_cluster_ids = _ordered_cluster_ids(
        candidates_by_cluster,
        cluster_order=cluster_order,
    )
    for cid in ordered_cluster_ids:
        cands = candidates_by_cluster[cid]
        vars_for_cluster = []
        for index, _cand in enumerate(cands):
            var = model.NewBoolVar(f"pick[{cid},{index}]")
            xvars[(cid, index)] = var
            vars_for_cluster.append(var)
        model.AddExactlyOne(vars_for_cluster)
        ordered_pick_vars.extend(vars_for_cluster)

    for a_key, b_key in sorted(incompatible):
        model.Add(xvars[a_key] + xvars[b_key] <= 1)

    objective_terms: list[Any] = []
    for cid in ordered_cluster_ids:
        for index, cand in enumerate(candidates_by_cluster[cid]):
            objective_terms.append(
                _candidate_effective_score(cand, candidate_bonus_by_key)
                * xvars[(cid, index)]
            )

    for (a_key, b_key), bonus in pair_bonus.items():
        za = xvars[a_key]
        zb = xvars[b_key]
        z = model.NewBoolVar(f"pair[{a_key[0]},{a_key[1]}|{b_key[0]},{b_key[1]}]")
        model.Add(z <= za)
        model.Add(z <= zb)
        model.Add(z >= za + zb - 1)
        objective_terms.append(int(bonus) * z)

    model.Maximize(sum(objective_terms))

    add_decision_strategy = getattr(model, "AddDecisionStrategy", None)
    choose_first = getattr(cp_model, "CHOOSE_FIRST", None)
    select_max_value = getattr(cp_model, "SELECT_MAX_VALUE", None)
    if (
        callable(add_decision_strategy)
        and choose_first is not None
        and select_max_value is not None
        and ordered_pick_vars
    ):
        add_decision_strategy(ordered_pick_vars, choose_first, select_max_value)

    pool: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    statuses: list[str] = []
    solve_limit = max(1, int(max_solutions))
    per_solve_time = max(0.5, float(time_limit_s) / max(1, min(solve_limit, 4)))

    for _idx in range(solve_limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = per_solve_time
        solver.parameters.num_search_workers = int(max(1, num_workers))
        solver.parameters.random_seed = 42

        status = solver.Solve(model)
        status_name = (
            solver.StatusName(status)
            if callable(getattr(solver, "StatusName", None))
            else str(status)
        )
        statuses.append(status_name)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        selected_keys: list[tuple[str, int]] = []
        for cid in ordered_cluster_ids:
            for index, _cand in enumerate(candidates_by_cluster[cid]):
                key = (cid, index)
                if solver.Value(xvars[key]) == 1:
                    selected_keys.append(key)
                    break

        if len(selected_keys) != len(ordered_cluster_ids):
            break

        transforms, selected_variants, chosen = _selected_from_solver(
            candidates_by_cluster,
            xvars,
            solver,
        )
        signature = _solution_signature(transforms, selected_variants)
        if signature not in seen_signatures:
            verify = _verify_complete(
                tools,
                room_model=room_model,
                clusters_outlines=clusters_outlines,
                relation_plan=relation_plan,
                grid_mm=grid_mm,
                cluster_transforms=transforms,
                selected_variants=selected_variants,
            )
            if bool(verify.get("hard_valid")) and bool(verify.get("complete")):
                pool.append(
                    {
                        "cluster_transforms": transforms,
                        "selected_variants": selected_variants,
                        "chosen": chosen,
                        "verify": verify,
                        "objective_value": int(round(solver.ObjectiveValue())),
                        "signature": signature,
                    }
                )
                seen_signatures.add(signature)

        model.Add(sum(xvars[key] for key in selected_keys) <= len(selected_keys) - 1)

    debug = {
        "incompatible_pair_count": len(incompatible),
        "pair_bonus_count": len(pair_bonus),
        "solver_statuses": statuses,
        "pool_size": len(pool),
    }
    return pool, debug


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


def _cluster_can_be_skipped_without_candidates(
    *,
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> bool:
    if not isinstance(relation_plan, Mapping):
        return True
    if _cluster_is_core_or_protected(relation_plan, cluster_id):
        return False
    return _cluster_priority_kind(relation_plan, cluster_id) in {"support", "optional"}


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
        cid
        for cid, candidates in candidates_by_cluster.items()
        if not candidates
        and _cluster_can_be_skipped_without_candidates(
            relation_plan=relation_plan,
            cluster_id=cid,
        )
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
    macro_ready = _verify_macro_ready(verify)
    micro_ready = acceptable

    return {
        "phase": "micro_repair" if macro_ready else "macro_blocked",
        "ready": macro_ready,
        "needed": bool(macro_ready and not acceptable),
        "seed_kind": (
            "acceptable_valid"
            if acceptable
            else "macro_ready_complete"
            if macro_ready
            else "hard_valid_complete"
        ),
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
            "macro_ready": macro_ready,
            "micro_ready": micro_ready,
            "macro_penalty_mm": int(quality.get("macro_penalty_mm") or 0),
            "micro_penalty_mm": int(quality.get("micro_penalty_mm") or 0),
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
                "anchor_front_matches_pair_direction",
                "primary_pair_axis_aligned",
                "support_band_consistency",
                "secondary_seat_faces_cluster",
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


def _solution_record_from_pool_entry(
    *,
    pool_entry: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    solution_index: int,
) -> dict[str, Any]:
    chosen_raw = pool_entry.get("chosen")
    chosen = chosen_raw if isinstance(chosen_raw, Mapping) else {}
    verify_raw = pool_entry.get("verify")
    verify = verify_raw if isinstance(verify_raw, Mapping) else {}
    cluster_transforms = [
        dict(item)
        for item in pool_entry.get("cluster_transforms", [])
        if isinstance(item, dict)
    ]
    selected_variants = [
        dict(item)
        for item in pool_entry.get("selected_variants", [])
        if isinstance(item, dict)
    ]
    quality = _global_quality_scores(
        verify=verify,
        relation_plan=relation_plan,
        chosen=chosen,
    )
    diagnostics = _solution_diagnostics(
        verify=verify,
        relation_plan=relation_plan,
        chosen=chosen,
    )
    record = {
        "solution_id": f"sol_{solution_index:02d}",
        "quality_rank": solution_index,
        "hard_valid": bool(verify.get("hard_valid")),
        "selected_variants": selected_variants,
        "cluster_transforms": cluster_transforms,
        "global_quality": quality,
        "diagnostics": diagnostics,
        "verify_summary": {
            "state_signature": str(verify.get("state_signature") or ""),
            "layout_score": _as_int(_quality_dict(verify).get("layout_score")),
            "macro_penalty_mm": _as_int(_quality_dict(verify).get("macro_penalty_mm")),
            "micro_penalty_mm": _as_int(_quality_dict(verify).get("micro_penalty_mm")),
            "quality_gate_reasons": list(
                (verify.get("quality_gate") or {}).get("reasons") or []
            )
            if isinstance(verify.get("quality_gate"), Mapping)
            else [],
        },
        "_rank_key": (
            quality["total_score"],
            quality["functionality_score"],
            quality["semantic_coherence_score"],
            _as_int(pool_entry.get("objective_value")),
        ),
        "_verify": dict(verify),
        "_chosen": dict(chosen),
        "_signature": str(pool_entry.get("signature") or ""),
    }
    violations = _solution_publishability_violations(
        record,
        relation_plan=relation_plan,
    )
    record["publishable"] = not violations
    record["publishability_reasons"] = violations
    return record


def _rank_solution_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_tuple(item: Mapping[str, Any]) -> tuple[Any, ...]:
        explicit = item.get("_rank_key")
        if explicit:
            return tuple(explicit)
        quality = item.get("global_quality")
        quality_map = quality if isinstance(quality, Mapping) else {}
        diagnostics = item.get("diagnostics")
        diagnostics_map = diagnostics if isinstance(diagnostics, Mapping) else {}
        return (
            _as_float(quality_map.get("total_score")),
            _as_float(quality_map.get("functionality_score")),
            _as_float(quality_map.get("semantic_coherence_score")),
            1 if diagnostics_map.get("dominant_anchor_correct") else 0,
        )

    ranked = sorted(
        records,
        key=lambda item: (
            bool(item.get("hard_valid")),
            rank_tuple(item),
            -_as_int((item.get("verify_summary") or {}).get("macro_penalty_mm")),
            -_as_int((item.get("verify_summary") or {}).get("micro_penalty_mm")),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        signature = str(item.get("_signature") or "")
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        clean = {
            key: value
            for key, value in item.items()
            if key not in {"_rank_key", "_verify", "_chosen", "_signature"}
        }
        clean["quality_rank"] = len(out) + 1
        clean["solution_id"] = f"sol_{len(out) + 1:02d}"
        out.append(clean)
    return out


def _best_solution_debug_record(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Candidate]]:
    if not records:
        return None, {}
    ranked = sorted(
        records,
        key=lambda item: tuple(item.get("_rank_key") or ()),
        reverse=True,
    )
    best = ranked[0]
    verify_raw = best.get("_verify")
    chosen_raw = best.get("_chosen")
    verify = verify_raw if isinstance(verify_raw, dict) else None
    chosen = chosen_raw if isinstance(chosen_raw, dict) else {}
    return verify, chosen


def solve_layout(
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    cluster_constraints: Any | None = None,
    grid_mm: int,
    tools_path: str | os.PathLike[str],
    max_variants_per_cluster: int = DEFAULT_MAX_VARIANTS_PER_CLUSTER,
    initial_candidates_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT,
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS,
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT,
    num_workers: int = 8,
    max_feasible_solutions_per_concept: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT,
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
    clusters_c = _filter_clusters_outlines_by_concept_variant_policy(
        clusters_c,
        relation_plan_aug,
    )

    variant_payload_lookup = _build_variant_payload_lookup(
        tools,
        clusters_c,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    candidates_by_cluster, cluster_order = _enumerate_initial_candidates(
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
    candidates_by_cluster = _filter_candidates_by_concept_variant_policy(
        candidates_by_cluster,
        relation_plan_aug,
    )
    candidates_by_cluster = _filter_protected_topology_candidates(
        candidates_by_cluster,
        relation_plan_aug,
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
    cluster_ids = _ordered_cluster_ids(
        candidates_by_cluster,
        cluster_order=cluster_order,
    )
    required_clusters_without_candidates = [
        cluster_id
        for cluster_id in cluster_ids
        if not candidates_by_cluster.get(cluster_id)
    ]
    if required_clusters_without_candidates:
        return {
            "status": "UNSAT",
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
            "solutions": [],
            "best_solution_id": None,
            "grid_mm": grid_mm,
            "cluster_transforms": [],
            "selected_variants": [],
            "global_quality": {},
            "diagnostics": {},
            "degradation_applied": [],
            "missing": [],
            "conflicts": [
                "Required clusters have no concept-compatible hard-valid candidates."
            ],
            "notes": [
                "Candidate generation rejected all variants for required clusters: "
                f"{required_clusters_without_candidates}",
            ],
            "skipped_clusters": skipped_candidate_clusters,
            "placer_seed": {
                "phase": "macro_blocked",
                "ready": False,
                "needed": False,
                "reason": "required_clusters_without_candidates",
                "required_clusters_without_candidates": (
                    required_clusters_without_candidates
                ),
            },
            "solver_debug": {
                "candidate_counts": {
                    cid: len(vals) for cid, vals in candidates_by_cluster.items()
                },
                "cluster_constraints_summary": _cluster_constraints_summary(
                    cluster_constraints_n
                ),
            },
        }
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

    solution_records: list[dict[str, Any]] = []
    best_verify: dict[str, Any] | None = None
    best_transforms: list[dict[str, Any]] = []
    best_variants: list[dict[str, str]] = []
    best_chosen: dict[str, Candidate] = {}
    used_relaxed_candidate_expansion = False
    infeasible_candidate_set = False
    wall_pin_focus = True
    solver_pool_debug: list[dict[str, Any]] = []
    root_strategy_debug: dict[str, Any] = {
        "root_cluster_id": _select_root_cluster_id(
            candidates_by_cluster,
            relation_plan=relation_plan_aug,
            cluster_order=cluster_ids,
        ),
        "cluster_order": list(cluster_ids),
        "used_wall_pin_focus": True,
        "wall_pin_focus_requested": True,
    }

    current_candidates = {
        cid: list(vals) for cid, vals in candidates_by_cluster.items()
    }
    current_candidates = _cap_candidate_search_space(current_candidates)

    for round_idx in range(max_rounds):
        logger.info(
            "Solver round %s/%s: candidate_counts=%s",
            round_idx + 1,
            max_rounds,
            {cid: len(vals) for cid, vals in current_candidates.items()},
        )
        round_candidates, round_cluster_ids, candidate_bonus_by_key, round_strategy = (
            _prepare_round_candidates(
                current_candidates,
                relation_plan=relation_plan_aug,
                cluster_order=cluster_ids,
                wall_pin_focus=wall_pin_focus,
            )
        )
        root_strategy_debug = round_strategy
        try:
            pool, pool_debug = _solve_solution_pool_one_round(
                tools=tools,
                room_model=room_model_c,
                clusters_outlines=clusters_c,
                relation_plan=relation_plan_aug,
                grid_mm=grid_mm,
                candidates_by_cluster=round_candidates,
                cluster_order=round_cluster_ids,
                variant_payload_lookup=variant_payload_lookup,
                time_limit_s=time_limit_s,
                num_workers=num_workers,
                candidate_bonus_by_key=candidate_bonus_by_key,
                max_solutions=max_feasible_solutions_per_concept,
            )
            solver_pool_debug.append(
                {
                    "round": round_idx + 1,
                    **pool_debug,
                    "candidate_counts": {
                        cid: len(vals) for cid, vals in round_candidates.items()
                    },
                }
            )
        except RuntimeError as exc:
            if bool(round_strategy.get("used_wall_pin_focus")):
                if not used_relaxed_candidate_expansion:
                    logger.info(
                        "Solver root wall-pin focus was infeasible at round %s/%s; "
                        "expanding candidates while preserving planner wall focus.",
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
                        cluster_order=cluster_ids,
                    )
                    current_candidates = _cap_candidate_search_space(current_candidates)
                    used_relaxed_candidate_expansion = True
                    continue
                logger.warning(
                    "Solver exact solve remained infeasible under mandatory wall-pin focus: %s",
                    exc,
                )
                infeasible_candidate_set = True
                break
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
                    cluster_order=cluster_ids,
                )
                current_candidates = _cap_candidate_search_space(current_candidates)
                used_relaxed_candidate_expansion = True
                continue
            logger.warning("Solver exact solve remained infeasible: %s", exc)
            infeasible_candidate_set = True
            break

        if not pool:
            if not used_relaxed_candidate_expansion:
                logger.info(
                    "Solver found no complete hard-valid solution at round %s/%s; "
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
                    cluster_order=cluster_ids,
                )
                current_candidates = _cap_candidate_search_space(current_candidates)
                used_relaxed_candidate_expansion = True
                continue
            infeasible_candidate_set = True
            break

        for entry in pool:
            solution_records.append(
                _solution_record_from_pool_entry(
                    pool_entry=entry,
                    relation_plan=relation_plan_aug,
                    solution_index=len(solution_records) + 1,
                )
            )

        best_round_records = sorted(
            solution_records,
            key=lambda item: tuple(item.get("_rank_key") or ()),
            reverse=True,
        )
        if best_round_records:
            best_record = best_round_records[0]
            best_verify_raw = best_record.get("_verify")
            best_chosen_raw = best_record.get("_chosen")
            best_verify = best_verify_raw if isinstance(best_verify_raw, dict) else None
            best_chosen = best_chosen_raw if isinstance(best_chosen_raw, dict) else {}
            best_transforms = [
                dict(item)
                for item in best_record.get("cluster_transforms", [])
                if isinstance(item, dict)
            ]
            best_variants = [
                dict(item)
                for item in best_record.get("selected_variants", [])
                if isinstance(item, dict)
            ]

        publishable_records = [
            item
            for item in solution_records
            if _is_publishable_solution(item, relation_plan=relation_plan_aug)
        ]
        if (
            publishable_records
            and len(solution_records) >= max_feasible_solutions_per_concept
        ):
            logger.info(
                "Solver collected %s feasible solutions including %s publishable "
                "solutions at round %s/%s.",
                len(solution_records),
                len(publishable_records),
                round_idx + 1,
                max_rounds,
            )
            break

        repair = (
            best_verify.get("repair_guidance") if isinstance(best_verify, dict) else {}
        )
        if not isinstance(repair, Mapping):
            repair = {}
        problem_clusters = [
            x.get("cluster_id")
            for x in (repair.get("prioritized_clusters") or [])
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        ]
        problem_clusters = _primary_seed_problem_clusters(
            problem_clusters,
            cluster_ids=cluster_ids,
            max_clusters=2,
        )

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
            chosen=best_chosen,
            problematic_cluster_ids=problem_clusters,
            max_variants_per_cluster=max_variants_per_cluster,
            cluster_allowed_rotations=cluster_allowed_rots,
            add_limit_per_cluster=_round_candidate_growth_limit(
                initial_candidates_per_cluster
            ),
        )
        # Keep the search exact but bounded.
        for cid, vals in current_candidates.items():
            current_candidates[cid] = _dedupe_candidates(vals)[
                : _expanded_candidate_limit(initial_candidates_per_cluster)
            ]
        current_candidates = _cap_candidate_search_space(current_candidates)

    publishable_solution_records = [
        item
        for item in solution_records
        if _is_publishable_solution(item, relation_plan=relation_plan_aug)
    ]
    ranked_solutions = _rank_solution_records(publishable_solution_records)[
        :DEFAULT_MAX_CROSS_CONCEPT_FINALISTS
    ]
    all_ranked_solutions = _rank_solution_records(solution_records)[
        :DEFAULT_MAX_CROSS_CONCEPT_FINALISTS
    ]
    if ranked_solutions:
        best_solution = ranked_solutions[0]
        best_solution_id = str(best_solution.get("solution_id") or "sol_01")
        best_transforms = [
            dict(item)
            for item in best_solution.get("cluster_transforms", [])
            if isinstance(item, dict)
        ]
        best_variants = [
            dict(item)
            for item in best_solution.get("selected_variants", [])
            if isinstance(item, dict)
        ]
        best_signature = _solution_signature(best_transforms, best_variants)
        for record in solution_records:
            if str(record.get("_signature") or "") != best_signature:
                continue
            best_verify_raw = record.get("_verify")
            best_chosen_raw = record.get("_chosen")
            best_verify = best_verify_raw if isinstance(best_verify_raw, dict) else None
            best_chosen = best_chosen_raw if isinstance(best_chosen_raw, dict) else {}
            break
        candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
        status = "OK"
        degradation_applied = (
            [
                {
                    "type": "skip_clusters_without_candidates",
                    "cluster_ids": skipped_candidate_clusters,
                }
            ]
            if skipped_candidate_clusters
            else []
        )
        return {
            "status": status,
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
            "solutions": ranked_solutions,
            "best_solution_id": best_solution_id,
            "grid_mm": grid_mm,
            "cluster_transforms": best_transforms,
            "selected_variants": best_variants,
            "global_quality": best_solution.get("global_quality") or {},
            "diagnostics": best_solution.get("diagnostics") or {},
            "degradation_applied": degradation_applied,
            "missing": [],
            "conflicts": [],
            "notes": [
                *(
                    [
                        "Skipped clusters without hard-valid candidates: "
                        f"{skipped_candidate_clusters}"
                    ]
                    if skipped_candidate_clusters
                    else []
                ),
                "Solved with quality-aware finite-pose CP-SAT over enumerated candidates.",
                "Filtered hard-valid solutions through the publishable gate before reranking.",
                f"macro_root_cluster={root_strategy_debug.get('root_cluster_id')}",
                f"root_wall_pin_focus_used={bool(root_strategy_debug.get('used_wall_pin_focus'))}",
                f"solution_pool_size={len(ranked_solutions)}",
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
                solver_status=status,
                candidate_counts=candidate_counts,
            ),
            "solver_debug": {
                "best_verify": best_verify,
                "candidate_counts": candidate_counts,
                "macro_strategy": root_strategy_debug,
                "solution_pool_debug": solver_pool_debug,
                "unpublishable_solution_count": max(
                    0,
                    len(solution_records) - len(publishable_solution_records),
                ),
                "cluster_constraints_summary": _cluster_constraints_summary(
                    cluster_constraints_n
                ),
            },
        }

    if all_ranked_solutions:
        candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
        reject_reasons = sorted(
            {
                reason
                for item in solution_records
                for reason in _solution_publishability_violations(
                    item,
                    relation_plan=relation_plan_aug,
                )
            }
        )
        notes = [
            "Found complete hard-valid layouts, but none passed the publishable gate.",
            "Returning HARD_VALID_BUT_UNACCEPTABLE instead of publishing a macro-bad layout.",
            f"publishability_reasons={json.dumps(reject_reasons, ensure_ascii=False)}",
            f"macro_root_cluster={root_strategy_debug.get('root_cluster_id')}",
            f"root_wall_pin_focus_used={bool(root_strategy_debug.get('used_wall_pin_focus'))}",
            f"hard_valid_solution_pool_size={len(all_ranked_solutions)}",
        ]
        return {
            "status": "HARD_VALID_BUT_UNACCEPTABLE",
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
            "solutions": [],
            "best_solution_id": None,
            "grid_mm": grid_mm,
            "cluster_transforms": [],
            "selected_variants": [],
            "global_quality": {},
            "diagnostics": {},
            "degradation_applied": [],
            "missing": [],
            "conflicts": notes,
            "notes": notes,
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
                solver_status="HARD_VALID_BUT_UNACCEPTABLE",
                candidate_counts=candidate_counts,
            ),
            "solver_debug": {
                "best_verify": best_verify,
                "best_transforms": best_transforms,
                "best_selected_variants": best_variants,
                "candidate_counts": candidate_counts,
                "macro_strategy": root_strategy_debug,
                "solution_pool_debug": solver_pool_debug,
                "unpublishable_solutions": all_ranked_solutions,
                "publishability_reasons": reject_reasons,
                "cluster_constraints_summary": _cluster_constraints_summary(
                    cluster_constraints_n
                ),
            },
        }

    notes = [
        "No complete hard-valid solution found within the finite candidate space.",
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
    notes.append(f"macro_root_cluster={root_strategy_debug.get('root_cluster_id')}")
    notes.append(
        f"root_wall_pin_focus_used={bool(root_strategy_debug.get('used_wall_pin_focus'))}"
    )
    if best_verify is not None:
        reasons = (best_verify.get("quality_gate") or {}).get("reasons") or []
        notes.append(f"quality_gate_reasons={json.dumps(reasons, ensure_ascii=False)}")
        notes.append(f"state_signature={best_verify.get('state_signature')}")
        notes.append(
            f"best_layout_score={_as_int(_quality_dict(best_verify).get('layout_score'))}"
        )
    candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
    return {
        "status": "UNSAT",
        "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
        "solutions": [],
        "best_solution_id": None,
        "grid_mm": grid_mm,
        "cluster_transforms": [],
        "selected_variants": [],
        "global_quality": {},
        "diagnostics": {},
        "degradation_applied": [],
        "missing": [],
        "conflicts": notes,
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
            "macro_strategy": root_strategy_debug,
            "solution_pool_debug": solver_pool_debug,
            "cluster_constraints_summary": _cluster_constraints_summary(
                cluster_constraints_n
            ),
        },
    }


def _normalize_point_list(points: Any) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        try:
            if isinstance(point, Mapping):
                x = point.get("x")
                y = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x = point[0]
                y = point[1]
            else:
                continue
            out.append({"x": int(round(float(x))), "y": int(round(float(y)))})
        except (TypeError, ValueError):
            continue
    return out


def _normalize_polygon_list(polygons: Any) -> list[list[dict[str, int]]]:
    if not isinstance(polygons, list):
        return []
    out: list[list[dict[str, int]]] = []
    for polygon in polygons:
        normalized = _normalize_point_list(polygon)
        if normalized:
            out.append(normalized)
    return out


def _normalize_room_model_for_tools(room_model: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(room_model)
    room = out.get("room")
    if isinstance(room, Mapping) and isinstance(room.get("polygon_ccw"), list):
        return out

    polygon = _normalize_point_list(
        room_model.get("polygon_mm") or room_model.get("polygon_ccw")
    )
    if polygon:
        out["room"] = {
            **(dict(room) if isinstance(room, Mapping) else {}),
            "polygon_ccw": polygon,
        }
    return out


def _cluster_payload_from_variant_bundle(
    row: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    cluster_id = str(row.get("cluster_id") or "").strip()
    variants = row.get("variant_bundle")
    if not cluster_id or not isinstance(variants, list) or not variants:
        return None
    first_variant = next((item for item in variants if isinstance(item, dict)), None)
    if first_variant is None:
        return None

    outline_polygons = _normalize_polygon_list(
        first_variant.get("interaction_hull_polygons_mm")
    ) or _normalize_polygon_list(first_variant.get("tight_hull_polygons_mm"))
    outline = _normalize_point_list(
        first_variant.get("interaction_hull_polygon_mm")
        or first_variant.get("tight_hull_polygon_mm")
    )
    if outline and not outline_polygons:
        outline_polygons = [outline]
    rects = [
        dict(item)
        for item in first_variant.get("interaction_placements")
        or first_variant.get("local_placements")
        or []
        if isinstance(item, Mapping)
    ]
    if not outline_polygons and not rects:
        return None
    xs = [
        point["x"]
        for polygon in outline_polygons
        for point in polygon
        if isinstance(point, Mapping)
    ]
    ys = [
        point["y"]
        for polygon in outline_polygons
        for point in polygon
        if isinstance(point, Mapping)
    ]
    if not xs or not ys:
        xs = []
        ys = []
        for rect in rects:
            try:
                x = int(round(float(rect.get("x", 0))))
                y = int(round(float(rect.get("y", 0))))
                w = int(round(float(rect.get("w", 0))))
                h = int(round(float(rect.get("h", 0))))
            except (TypeError, ValueError):
                continue
            xs.extend([x, x + max(w, 0)])
            ys.extend([y, y + max(h, 0)])
    if not xs or not ys:
        return None
    payload = {
        "status": "OK",
        "cluster_id": cluster_id,
        "local_frame": {
            "unit": "mm",
            "grid_mm": DEFAULT_GRID_MM,
            "origin_note": "(0,0) is an arbitrary local origin for this cluster",
        },
        "local_placements": deepcopy(first_variant.get("local_placements") or []),
        "cluster_footprint": {
            "type": "union_of_rects",
            "rects": [],
            "local_bbox": {
                "min_x": min(xs),
                "min_y": min(ys),
                "max_x": max(xs),
                "max_y": max(ys),
            },
            "outline_polygons_ccw": outline_polygons,
        },
        "variant_bundle": [dict(item) for item in variants if isinstance(item, dict)],
        "notes": [],
        "missing": [],
        "conflicts": [],
    }
    return cluster_id, payload


def _clusters_outlines_from_variant_bundles(
    cluster_variant_bundles: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(cluster_variant_bundles, list):
        return out
    for row in cluster_variant_bundles:
        if not isinstance(row, Mapping):
            continue
        converted = _cluster_payload_from_variant_bundle(row)
        if converted is None:
            continue
        cluster_id, payload = converted
        out[cluster_id] = payload
    return out


def _relation_plan_for_macro_concept(
    *,
    concept: Mapping[str, Any],
    room_model: Mapping[str, Any],
    semantic_layout_program: Mapping[str, Any],
) -> dict[str, Any]:
    explicit = concept.get("solver_relation_plan")
    if isinstance(explicit, dict):
        out = dict(explicit)
        out["macro_concept"] = {
            key: value
            for key, value in concept.items()
            if key != "solver_relation_plan"
        }
        return out

    relation_keys = {
        "cluster_affinities",
        "cluster_orientations",
        "object_orientations",
        "cluster_relations",
        "cluster_directional_relations",
        "circulation_plan",
        "layout_intent_profile",
    }
    if any(key in concept for key in relation_keys):
        out = {
            key: deepcopy(concept.get(key)) for key in relation_keys if key in concept
        }
        out["status"] = "OK"
        out["macro_concept"] = {
            key: value for key, value in concept.items() if key not in relation_keys
        }
        return out

    from agent.seed_concept_generator import solver_plan_from_concept

    return solver_plan_from_concept(
        concept=dict(concept),
        room_model_json=dict(room_model),
        room_type=str(semantic_layout_program.get("room_type") or ""),
    )


def _macro_concepts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    bundle = payload.get("macro_concept_bundle")
    if not isinstance(bundle, Mapping):
        return []
    concepts = bundle.get("concepts")
    if not isinstance(concepts, list):
        return []
    out = [dict(item) for item in concepts if isinstance(item, Mapping)]
    return out[:DEFAULT_MAX_CONCEPTS]


def _inventory_cluster_constraints(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    inventory = payload.get("inventory_decision_program")
    if isinstance(inventory, Mapping) and isinstance(inventory.get("clusters"), list):
        return dict(inventory)
    semantic = payload.get("semantic_layout_program")
    if isinstance(semantic, Mapping) and isinstance(semantic.get("clusters"), list):
        return dict(semantic)
    return None


def _primary_cluster_ids_from_concepts(
    concepts: Sequence[Mapping[str, Any]],
) -> set[str]:
    protected: set[str] = set()
    for concept in concepts:
        for row in concept.get("cluster_zone_plan") or []:
            if not isinstance(row, Mapping):
                continue
            role = str(row.get("role") or row.get("cluster_role") or "").lower()
            claim = str(row.get("anchor_role") or row.get("wall_claim") or "").lower()
            cluster_id = str(row.get("cluster_id") or "").strip()
            if cluster_id and (
                "primary" in role
                or "dominant" in role
                or "anchor" in role
                or claim == "strong"
            ):
                protected.add(cluster_id)
    return protected


def _controlled_degradation_candidates(
    *,
    payload: Mapping[str, Any],
    concepts: Sequence[Mapping[str, Any]],
    available_cluster_ids: Sequence[str],
) -> list[str]:
    protected = _primary_cluster_ids_from_concepts(concepts)
    scores: dict[str, tuple[int, str]] = {}

    def consider(cluster_id: str, marker: str, priority: int) -> None:
        clean_id = str(cluster_id or "").strip()
        if (
            not clean_id
            or clean_id in protected
            or clean_id not in available_cluster_ids
        ):
            return
        current = scores.get(clean_id)
        item = (priority, marker)
        if current is None or item > current:
            scores[clean_id] = item

    semantic = payload.get("semantic_layout_program")
    if isinstance(semantic, Mapping):
        for row in semantic.get("active_clusters") or []:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            text = json.dumps(row, sort_keys=True).lower()
            if "optional" in text:
                consider(cluster_id, "semantic_optional", 30)
            elif "support" in text:
                consider(cluster_id, "semantic_support", 20)
            elif "low" in text:
                consider(cluster_id, "semantic_low_priority", 10)

    inventory = payload.get("inventory_decision_program")
    if isinstance(inventory, Mapping):
        for row in inventory.get("cluster_decisions") or []:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            text = json.dumps(row, sort_keys=True).lower()
            if "optional" in text:
                consider(cluster_id, "inventory_optional", 30)
            elif "support" in text:
                consider(cluster_id, "inventory_support", 20)
            elif "low" in text:
                consider(cluster_id, "inventory_low_priority", 10)

    return [
        cluster_id
        for cluster_id, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1][0], item[0]),
        )
    ]


def solve_global_layout_bundle(
    *,
    payload: dict[str, Any],
    grid_mm: int,
    tools_path: str | os.PathLike[str],
    max_variants_per_cluster: int = DEFAULT_MAX_VARIANTS_PER_CLUSTER,
    initial_candidates_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT,
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS,
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT,
    num_workers: int = 8,
    max_feasible_solutions_per_concept: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT,
) -> dict[str, Any]:
    required = [
        "room_model",
        "semantic_layout_program",
        "cluster_variant_bundles",
        "macro_concept_bundle",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": missing,
            "conflicts": [],
            "notes": ["Missing required solver input."],
        }

    room_model_raw = payload.get("room_model")
    semantic_raw = payload.get("semantic_layout_program")
    if not isinstance(room_model_raw, Mapping) or not isinstance(semantic_raw, Mapping):
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": [],
            "conflicts": ["room_model and semantic_layout_program must be objects"],
            "notes": [],
        }

    room_model = _normalize_room_model_for_tools(room_model_raw)
    concepts = _macro_concepts(payload)
    if not concepts:
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": ["macro_concept_bundle.concepts"],
            "conflicts": [],
            "notes": ["No macro concepts were provided."],
        }

    base_clusters = _clusters_outlines_from_variant_bundles(
        payload.get("cluster_variant_bundles")
    )
    if not base_clusters:
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": ["cluster_variant_bundles"],
            "conflicts": [],
            "notes": ["No hard-valid cluster variant bundles were usable."],
        }

    available_cluster_ids = list(base_clusters.keys())
    degradation_order = _controlled_degradation_candidates(
        payload=payload,
        concepts=concepts,
        available_cluster_ids=available_cluster_ids,
    )
    applied_degradations: list[dict[str, Any]] = []
    dropped_ids: set[str] = set()
    concept_outputs: list[dict[str, Any]] = []
    all_solutions: list[dict[str, Any]] = []
    hard_valid_unacceptable_count = 0

    for degradation_round in range(max(1, int(max_rounds)) + 1):
        clusters_for_round = _filter_cluster_ids_from_outlines(
            base_clusters, dropped_ids
        )
        for concept in concepts:
            relation_plan = _relation_plan_for_macro_concept(
                concept=concept,
                room_model=room_model,
                semantic_layout_program=semantic_raw,
            )
            relation_plan = _filter_relation_plan_clusters(relation_plan, dropped_ids)
            if relation_plan is None:
                continue
            concept_result = solve_layout(
                room_model=room_model,
                clusters_outlines=clusters_for_round,
                relation_plan=relation_plan,
                cluster_constraints=_inventory_cluster_constraints(payload),
                grid_mm=int(grid_mm or DEFAULT_GRID_MM),
                tools_path=tools_path,
                max_variants_per_cluster=max_variants_per_cluster,
                initial_candidates_per_cluster=initial_candidates_per_cluster,
                max_rounds=max_rounds,
                time_limit_s=time_limit_s,
                num_workers=num_workers,
                max_feasible_solutions_per_concept=max_feasible_solutions_per_concept,
            )
            concept_outputs.append(concept_result)
            if str(concept_result.get("status") or "") == (
                "HARD_VALID_BUT_UNACCEPTABLE"
            ):
                debug = concept_result.get("solver_debug")
                debug_map = debug if isinstance(debug, Mapping) else {}
                hard_valid_unacceptable_count += max(
                    1,
                    len(debug_map.get("unpublishable_solutions") or []),
                )
            concept_id = str(concept_result.get("selected_concept_id") or "")
            for solution in concept_result.get("solutions") or []:
                if not isinstance(solution, dict):
                    continue
                if not bool(solution.get("publishable", True)):
                    continue
                enriched = dict(solution)
                enriched["concept_id"] = concept_id
                enriched["degradation_applied"] = list(applied_degradations)
                all_solutions.append(enriched)

        if all_solutions:
            break
        if degradation_round >= max(0, int(max_rounds)):
            break
        next_drop = next(
            (
                cluster_id
                for cluster_id in degradation_order
                if cluster_id not in dropped_ids
            ),
            None,
        )
        if next_drop is None:
            break
        dropped_ids.add(next_drop)
        applied_degradations.append(
            {
                "round": degradation_round + 1,
                "type": "drop_optional_or_support_cluster",
                "cluster_ids": [next_drop],
            }
        )

    ranked = _rank_solution_records(all_solutions)[:DEFAULT_MAX_CROSS_CONCEPT_FINALISTS]
    if ranked:
        best_solution = ranked[0]
        selected_concept_id = str(best_solution.get("concept_id") or "")
        return {
            "status": "OK",
            "selected_concept_id": selected_concept_id,
            "solutions": ranked,
            "best_solution_id": best_solution.get("solution_id"),
            "cluster_transforms": best_solution.get("cluster_transforms") or [],
            "selected_variants": best_solution.get("selected_variants") or [],
            "global_quality": best_solution.get("global_quality") or {},
            "diagnostics": best_solution.get("diagnostics") or {},
            "degradation_applied": applied_degradations,
            "missing": [],
            "conflicts": [],
            "notes": [
                "Compared publishable solution pools across macro concepts.",
                f"concepts_evaluated={len(concepts)}",
                f"solution_pool_size={len(ranked)}",
            ],
            "solver_debug": {
                "concept_statuses": [
                    {
                        "concept_id": item.get("selected_concept_id"),
                        "status": item.get("status"),
                        "solution_count": len(item.get("solutions") or []),
                    }
                    for item in concept_outputs
                ],
            },
        }

    if hard_valid_unacceptable_count > 0:
        return {
            "status": "HARD_VALID_BUT_UNACCEPTABLE",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "cluster_transforms": [],
            "selected_variants": [],
            "global_quality": {},
            "diagnostics": {},
            "degradation_applied": applied_degradations,
            "missing": [],
            "conflicts": [
                "Hard-valid solution pools existed, but none passed the publishable gate."
            ],
            "notes": [
                "Degradation rounds were attempted before returning non-publishable status.",
                f"concepts_evaluated={len(concepts)}",
                f"hard_valid_unacceptable_count={hard_valid_unacceptable_count}",
            ],
            "solver_debug": {
                "concept_statuses": [
                    {
                        "concept_id": item.get("selected_concept_id"),
                        "status": item.get("status"),
                        "solution_count": len(item.get("solutions") or []),
                    }
                    for item in concept_outputs
                ],
            },
        }

    return {
        "status": "UNSAT",
        "selected_concept_id": None,
        "solutions": [],
        "best_solution_id": None,
        "cluster_transforms": [],
        "selected_variants": [],
        "global_quality": {},
        "diagnostics": {},
        "degradation_applied": applied_degradations,
        "missing": [],
        "conflicts": [
            "No hard-valid solution found across the provided macro concepts."
        ],
        "notes": [
            "Exact finite-pose search exhausted the available concept candidate pools."
        ],
        "solver_debug": {
            "concept_statuses": [
                {
                    "concept_id": item.get("selected_concept_id"),
                    "status": item.get("status"),
                    "solution_count": len(item.get("solutions") or []),
                }
                for item in concept_outputs
            ],
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
