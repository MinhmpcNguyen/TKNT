from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Protocol

# type: ignore
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cluster_placer.tools_v2 import (  # type: ignore
    DiagnosePhase2Seed,
    EnumeratePhase2RepairMoves,
    PreviewPhase2Repair,
    ScorePhase2Repair,
)

try:
    from clients.openai_client import OpenAIClient  # type: ignore
except Exception:  # pragma: no cover
    OpenAIClient = None  # type: ignore

try:
    from prompt.cluster_placer import (
        MACRO_CLUSTER_PLACER_PROMPT as DEFAULT_PHASE2_PLACER_PROMPT,  # type: ignore
    )
except Exception:  # pragma: no cover
    DEFAULT_PHASE2_PLACER_PROMPT = "You are a phase-2 placer. Return JSON only."


AllowedRotation = Literal[0, 90, 180, 270]
RepairStatus = Literal["REPAIRED", "NO_IMPROVEMENT", "NEED_INFO"]
AllowedObjectOp = Literal[
    "rotate_object",
    "mirror_object",
    "nudge_object",
    "swap_objects",
    "set_anchor",
    "set_front_override",
]


class LLMCallable(Protocol):
    def __call__(
        self, *, system_prompt: str, user_payload_json: str
    ) -> str | Dict[str, Any]: ...


# -----------------------------------------------------------------------------
# Output schema
# -----------------------------------------------------------------------------
def _strip_required_str(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ClusterTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_id: str
    x: int
    y: int
    rot: AllowedRotation

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")


class ClusterVariantSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_id: str
    variant_id: str

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")

    @field_validator("variant_id")
    @classmethod
    def _variant_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "variant_id")


class ObjectRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_id: str
    object_id: str
    op: AllowedObjectOp
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")

    @field_validator("object_id")
    @classmethod
    def _object_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "object_id")


class Phase2RepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: RepairStatus
    cluster_transforms: List[ClusterTransform] = Field(default_factory=list)
    selected_variants: List[ClusterVariantSelection] = Field(default_factory=list)
    object_repairs: List[ObjectRepair] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def _notes_non_empty(cls, value: List[str]) -> List[str]:
        out: List[str] = []
        for item in value:
            out.append(_strip_required_str(item, "notes item"))
        return out

    @field_validator("cluster_transforms")
    @classmethod
    def _unique_cluster_transform_ids(
        cls, value: List[ClusterTransform]
    ) -> List[ClusterTransform]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_transforms cluster_id must be unique")
        return value

    @field_validator("selected_variants")
    @classmethod
    def _unique_selected_variant_ids(
        cls, value: List[ClusterVariantSelection]
    ) -> List[ClusterVariantSelection]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("selected_variants cluster_id must be unique")
        return value


# -----------------------------------------------------------------------------
# Payload helpers
# -----------------------------------------------------------------------------
def build_phase2_payload(
    room_01: Dict[str, Any],
    clusters_04: Dict[str, Any],
    outlines_05: Dict[str, Any],
    relation_05b: Dict[str, Any],
    placer_06: Dict[str, Any],
    cluster_constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from copy import deepcopy

    # -------------------------------------------------------------------------
    # local helpers
    # -------------------------------------------------------------------------
    def _normalize_vec(vec: tuple[float, float] | None) -> tuple[float, float] | None:
        if vec is None:
            return None
        x, y = float(vec[0]), float(vec[1])
        norm = (x * x + y * y) ** 0.5
        if norm <= 1e-9:
            return None
        return (x / norm, y / norm)

    def _parse_vec2(value: Any) -> tuple[float, float] | None:
        if isinstance(value, dict):
            dx = value.get("dx")
            dy = value.get("dy")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            dx, dy = value[0], value[1]
        else:
            return None
        try:
            return _normalize_vec((float(dx), float(dy)))
        except Exception:
            return None

    def _rotate_point_ccw_90s(x: float, y: float, rot: int) -> tuple[float, float]:
        r = int(rot) % 360
        if r == 0:
            return x, y
        if r == 90:
            return -y, x
        if r == 180:
            return -x, -y
        if r == 270:
            return y, -x
        raise ValueError(f"Unsupported rot={rot}")

    def _rotate_vec_ccw_90s(
        vec: tuple[float, float] | None, rot: int
    ) -> tuple[float, float] | None:
        if vec is None:
            return None
        return _normalize_vec(
            _rotate_point_ccw_90s(float(vec[0]), float(vec[1]), int(rot))
        )

    def _transform_rect_world(
        rect: Dict[str, Any], tx: int, ty: int, rot: int
    ) -> Dict[str, Any]:
        x = float(rect.get("x", 0))
        y = float(rect.get("y", 0))
        w = float(rect.get("w", 0))
        h = float(rect.get("h", 0))
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        world = [_rotate_point_ccw_90s(px, py, rot) for px, py in corners]
        world = [(px + tx, py + ty) for px, py in world]
        xs = [p[0] for p in world]
        ys = [p[1] for p in world]
        return {
            "polygon_ccw": [
                {"x": int(round(px)), "y": int(round(py))} for px, py in world
            ],
            "bbox": {
                "min_x": int(round(min(xs))),
                "min_y": int(round(min(ys))),
                "max_x": int(round(max(xs))),
                "max_y": int(round(max(ys))),
            },
            "world_center": {
                "x": int(round(sum(xs) / 4.0)),
                "y": int(round(sum(ys) / 4.0)),
            },
        }

    def _extract_room_context() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        room_model_used = seed.get("room_model_used")
        if isinstance(room_model_used, dict) and room_model_used:
            return {"room_model_used": deepcopy(room_model_used)}
        return {"room_model_used": deepcopy(room_01)}

    def _extract_phase_control() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        solver_debug = (placer_06 or {}).get("solver_debug") or {}
        best_verify = solver_debug.get("best_verify") or {}

        ready = bool(seed.get("ready"))
        needed = bool(seed.get("needed", True))
        seed_kind = str(seed.get("seed_kind") or "")
        if not seed_kind:
            if best_verify.get("hard_valid") and best_verify.get("complete"):
                seed_kind = "hard_valid_complete"
            elif best_verify.get("hard_valid"):
                seed_kind = "hard_valid_partial"
            else:
                seed_kind = "none"

        return {
            "phase": "micro_repair",
            "ready": ready or bool(best_verify),
            "needed": needed,
            "seed_kind": seed_kind,
            "solver_status": str((placer_06 or {}).get("status") or "UNKNOWN"),
        }

    def _extract_seed_layout() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        if isinstance(seed.get("seed_layout"), dict):
            data = seed["seed_layout"]
            return {
                "cluster_transforms": deepcopy(data.get("cluster_transforms") or []),
                "selected_variants": deepcopy(data.get("selected_variants") or []),
            }

        solver_debug = (placer_06 or {}).get("solver_debug") or {}
        return {
            "cluster_transforms": deepcopy(
                solver_debug.get("best_transforms")
                or (placer_06 or {}).get("cluster_transforms")
                or []
            ),
            "selected_variants": deepcopy(
                solver_debug.get("best_selected_variants")
                or (placer_06 or {}).get("selected_variants")
                or []
            ),
        }

    def _extract_goals() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        relation_plan_used = seed.get("relation_plan_used")
        cluster_constraints_used = seed.get("cluster_constraints_used")

        if not isinstance(relation_plan_used, dict) or not relation_plan_used:
            relation_plan_used = deepcopy(relation_05b)
        if not isinstance(cluster_constraints_used, dict):
            cluster_constraints_used = (
                deepcopy(cluster_constraints)
                if isinstance(cluster_constraints, dict)
                else None
            )

        return {
            "relation_plan_used": relation_plan_used,
            "cluster_constraints_used": cluster_constraints_used,
        }

    def _extract_repair_debug() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        solver_debug = (placer_06 or {}).get("solver_debug") or {}
        seed_verify = deepcopy(
            seed.get("seed_verify") or solver_debug.get("best_verify") or {}
        )
        return {
            "seed_metrics": deepcopy(seed.get("seed_metrics") or {}),
            "seed_verify": seed_verify,
            "repair_targets": deepcopy(
                seed.get("repair_targets") or seed_verify.get("repair_guidance") or {}
            ),
            "candidate_counts": deepcopy(
                seed.get("candidate_counts")
                or solver_debug.get("candidate_counts")
                or {}
            ),
            "quality_gate": deepcopy(seed_verify.get("quality_gate") or {}),
        }

    def _extract_edit_contract() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        contract = seed.get("edit_contract")
        if isinstance(contract, dict) and contract:
            return deepcopy(contract)
        return {
            "must_preserve_hard_valid": True,
            "hard_constraints_must_hold": [
                "out_of_bounds",
                "overlap",
                "door_swing",
            ],
            "primary_soft_targets": [
                "critical_orientation_penalty_mm",
                "focal_pair_penalty_mm",
                "max_critical_item_penalty_mm",
            ],
            "allowed_edit_levels": [
                "cluster_variant",
                "cluster_pose",
                "object_pose",
                "object_swap",
                "object_nudge",
            ],
        }

    def _index_clusters_by_id(
        clusters_json: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in (clusters_json or {}).get("clusters") or []:
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                out[row["cluster_id"]] = deepcopy(row)
        return out

    def _index_outlines_by_id() -> Dict[str, Dict[str, Any]]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        mat = seed.get("materialized_clusters_outlines")
        if isinstance(mat, dict) and mat:
            return deepcopy(mat)

        if isinstance(outlines_05, dict):
            # already keyed by cluster_id
            direct = {}
            for cid, row in outlines_05.items():
                if isinstance(cid, str) and isinstance(row, dict):
                    direct[cid] = deepcopy(row)
            if direct:
                return direct
            # or wrapped in {"clusters": [...]}
            if isinstance(outlines_05.get("clusters"), list):
                out = {}
                for row in outlines_05["clusters"]:
                    if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                        out[row["cluster_id"]] = deepcopy(row)
                return out
        return {}

    def _build_cluster_cards(seed_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        merged = _index_clusters_by_id(clusters_04)
        outlines = _index_outlines_by_id()

        active_ids = {
            str(x.get("cluster_id"))
            for x in seed_layout.get("cluster_transforms") or []
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        }
        if not active_ids:
            active_ids = set(merged.keys()) | set(outlines.keys())

        cards: List[Dict[str, Any]] = []
        for cid in sorted(active_ids):
            merged_row = merged.get(cid, {})
            outline_row = outlines.get(cid, {})
            if not merged_row and not outline_row:
                continue

            decisions = (
                merged_row.get("decisions")
                if isinstance(merged_row.get("decisions"), list)
                else []
            )

            object_sizes: Dict[str, Dict[str, int]] = {}
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                oid = d.get("id")
                dims = d.get("rep_dims_m")
                if isinstance(oid, str) and isinstance(dims, dict):
                    try:
                        object_sizes[oid] = {
                            "w_mm": int(round(float(dims.get("w", 0)) * 1000.0)),
                            "h_mm": int(round(float(dims.get("h", 0)) * 1000.0)),
                        }
                    except Exception:
                        pass

            # keep variants if present on either merged or outline side
            available_variants = []
            for key in ("available_variants", "variants"):
                src = None
                if isinstance(outline_row.get(key), list):
                    src = outline_row.get(key)
                elif isinstance(merged_row.get(key), list):
                    src = merged_row.get(key)
                if isinstance(src, list):
                    available_variants = deepcopy(src)
                    break

            cards.append(
                {
                    "cluster_id": cid,
                    "tag": merged_row.get("tag"),
                    "members": deepcopy(merged_row.get("members") or []),
                    "anchors": deepcopy(merged_row.get("anchors") or []),
                    "hard_constraints": deepcopy(
                        merged_row.get("hard_constraints") or []
                    ),
                    "soft_constraints": deepcopy(
                        merged_row.get("soft_constraints") or []
                    ),
                    "cluster_rules": deepcopy(merged_row.get("cluster_rules") or {}),
                    "decisions": deepcopy(decisions),
                    "object_sizes": object_sizes,
                    "local_placements": deepcopy(
                        outline_row.get("local_placements") or []
                    ),
                    "cluster_footprint": deepcopy(
                        outline_row.get("cluster_footprint") or {}
                    ),
                    "orientation_meta": deepcopy(
                        outline_row.get("orientation_meta") or {}
                    ),
                    "available_variants": available_variants,
                    "source": {
                        "merged_from_04": bool(merged_row),
                        "outline_from_05_or_solver": bool(outline_row),
                    },
                }
            )
        return cards

    def _top_object_debug(
        seed_verify: Dict[str, Any], max_items: int = 24
    ) -> Dict[tuple[str, str], Dict[str, Any]]:
        out: Dict[tuple[str, str], Dict[str, Any]] = {}
        quality = (seed_verify or {}).get("quality") or {}
        dbg = (quality.get("orientation_debug") or []) + (
            quality.get("critical_orientation_debug") or []
        )
        for item in dbg:
            if not isinstance(item, dict):
                continue
            cid = item.get("cluster_id")
            oid = item.get("object_id")
            if not isinstance(cid, str) or not isinstance(oid, str):
                continue
            key = (cid, oid)
            prev = out.get(key)
            pen = int(item.get("penalty_mm") or 0)
            if prev is None or pen > int(prev.get("penalty_mm") or 0):
                out[key] = {
                    "intent": item.get("intent"),
                    "penalty_mm": pen,
                    "front_clear_mm": item.get("front_clear_mm"),
                    "back_clear_mm": item.get("back_clear_mm"),
                    "best_clear_mm": item.get("best_clear_mm"),
                    "dot": item.get("dot"),
                }
        ranked = sorted(out.items(), key=lambda kv: -int(kv[1].get("penalty_mm") or 0))[
            :max_items
        ]
        return {k: v for k, v in ranked}

    def _build_objects_world(
        cluster_cards: List[Dict[str, Any]],
        seed_layout: Dict[str, Any],
        seed_verify: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tf_by_id = {
            str(x.get("cluster_id")): x
            for x in seed_layout.get("cluster_transforms") or []
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        }
        object_dbg = _top_object_debug(seed_verify)
        out: List[Dict[str, Any]] = []

        for card in cluster_cards:
            cid = str(card.get("cluster_id"))
            tf = tf_by_id.get(cid)
            if not isinstance(tf, dict):
                continue
            tx = int(tf.get("x") or 0)
            ty = int(tf.get("y") or 0)
            rot = int(tf.get("rot") or 0) % 360

            fp = card.get("cluster_footprint") or {}
            rects = fp.get("rects") or []
            important = (card.get("orientation_meta") or {}).get(
                "important_objects"
            ) or {}
            cluster_front = _parse_vec2(
                (card.get("orientation_meta") or {}).get("cluster_front_local")
            )

            for rect in rects:
                if not isinstance(rect, dict) or not isinstance(rect.get("id"), str):
                    continue
                oid = rect["id"]
                world_geom = _transform_rect_world(rect, tx, ty, rot)

                obj_meta = important.get(oid) if isinstance(important, dict) else {}
                front_local = (
                    _parse_vec2((obj_meta or {}).get("front_local")) or cluster_front
                )
                axis_local = _parse_vec2((obj_meta or {}).get("axis_local"))
                front_world = _rotate_vec_ccw_90s(front_local, rot)
                axis_world = _rotate_vec_ccw_90s(axis_local, rot)

                w = int(round(float(rect.get("w", 0))))
                h = int(round(float(rect.get("h", 0))))
                if front_local is None:
                    size_along_access = max(w, h)
                else:
                    dx, dy = abs(front_local[0]), abs(front_local[1])
                    size_along_access = w if dx >= dy else h
                required_clearance = int(round(0.25 * size_along_access))

                dbg = object_dbg.get((cid, oid), {})
                out.append(
                    {
                        "cluster_id": cid,
                        "object_id": oid,
                        "cluster_rot": rot,
                        "local_rect": {
                            "x": int(round(float(rect.get("x", 0)))),
                            "y": int(round(float(rect.get("y", 0)))),
                            "w": w,
                            "h": h,
                        },
                        "polygon_ccw": world_geom["polygon_ccw"],
                        "bbox": world_geom["bbox"],
                        "world_center": world_geom["world_center"],
                        "size_mm": {"w": w, "h": h},
                        "front_world": None
                        if front_world is None
                        else {
                            "dx": round(front_world[0], 3),
                            "dy": round(front_world[1], 3),
                        },
                        "axis_world": None
                        if axis_world is None
                        else {
                            "dx": round(axis_world[0], 3),
                            "dy": round(axis_world[1], 3),
                        },
                        "required_clearance_mm": required_clearance,
                        "current_front_clear_mm": dbg.get("front_clear_mm"),
                        "current_back_clear_mm": dbg.get("back_clear_mm"),
                        "best_clear_mm": dbg.get("best_clear_mm"),
                        "dominant_debug_intent": dbg.get("intent"),
                        "dominant_penalty_mm": dbg.get("penalty_mm"),
                        "dominant_dot": dbg.get("dot"),
                    }
                )

        out.sort(
            key=lambda x: (
                x["cluster_id"],
                -int(x.get("dominant_penalty_mm") or 0),
                x["object_id"],
            )
        )
        return out

    # -------------------------------------------------------------------------
    # build payload
    # -------------------------------------------------------------------------
    room_context = _extract_room_context()
    phase_control = _extract_phase_control()
    seed_layout = _extract_seed_layout()
    goals = _extract_goals()
    repair_debug = _extract_repair_debug()
    edit_contract = _extract_edit_contract()
    cluster_cards = _build_cluster_cards(seed_layout)
    objects_world = _build_objects_world(
        cluster_cards,
        seed_layout,
        repair_debug.get("seed_verify") or {},
    )

    return {
        "phase_control": phase_control,
        "room_context": room_context,
        "seed_layout": seed_layout,
        "cluster_cards": cluster_cards,
        "goals": goals,
        "repair_debug": repair_debug,
        "edit_contract": edit_contract,
        "objects_world": objects_world,
    }


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _cluster_cards_as_map(cluster_cards: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(cluster_cards, dict):
        return {
            str(k): deepcopy(v)
            for k, v in cluster_cards.items()
            if isinstance(k, str) and isinstance(v, dict)
        }
    if isinstance(cluster_cards, list):
        out: Dict[str, Dict[str, Any]] = {}
        for row in cluster_cards:
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                out[row["cluster_id"]] = deepcopy(row)
        return out
    return {}


def _seed_transform_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("cluster_transforms") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _seed_variant_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("selected_variants") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _allowed_object_ids_by_cluster(payload: Dict[str, Any]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for row in payload.get("objects_world") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        oid = row.get("object_id")
        if isinstance(cid, str) and isinstance(oid, str):
            out.setdefault(cid, set()).add(oid)
    return out


def _allowed_variant_ids_by_cluster(payload: Dict[str, Any]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    cards = _cluster_cards_as_map(payload.get("cluster_cards") or {})
    current_seed = _seed_variant_map(payload.get("seed_layout") or {})
    for cid, card in cards.items():
        allowed: set[str] = set()
        variants = card.get("available_variants") or card.get("variants") or []
        if isinstance(variants, list):
            for item in variants:
                if isinstance(item, dict) and isinstance(item.get("variant_id"), str):
                    allowed.add(item["variant_id"])
                elif isinstance(item, str):
                    allowed.add(item)
        if not allowed and cid in current_seed:
            allowed.add(str(current_seed[cid].get("variant_id") or ""))
        if allowed:
            out[cid] = allowed
    return out


def _allowed_cluster_rotations(payload: Dict[str, Any]) -> Dict[str, set[int]]:
    out: Dict[str, set[int]] = {}
    cards = _cluster_cards_as_map(payload.get("cluster_cards") or {})
    for cid, card in cards.items():
        vals: set[int] = set()
        rules = card.get("cluster_rules") or {}
        rots = rules.get("allowed_rotations") or []
        for r in rots:
            try:
                rv = int(r)
            except Exception:
                continue
            if rv in {0, 90, 180, 270}:
                vals.add(rv)
        if not vals:
            vals = {0, 90, 180, 270}
        out[cid] = vals
    return out


def _allowed_object_ops(payload: Dict[str, Any]) -> set[str]:
    contract = payload.get("edit_contract") or {}
    levels = contract.get("allowed_edit_levels") or {}
    allowed = set()
    if isinstance(levels, list):
        levels = {str(x): True for x in levels}
    if levels.get("object_pose", False) or levels.get("object_rotation", True):
        allowed.update({"rotate_object", "mirror_object", "set_front_override"})
    if levels.get("object_nudge", False) or levels.get("object_local_nudge", True):
        allowed.add("nudge_object")
    if levels.get("object_swap", False) or levels.get(
        "object_swap_within_cluster", True
    ):
        allowed.add("swap_objects")
    if levels.get("object_anchor", False):
        allowed.add("set_anchor")
    return allowed


# -----------------------------------------------------------------------------
# Compact payload + tool context
# -----------------------------------------------------------------------------
def prepare_phase2_llm_payload(
    full_payload: Dict[str, Any], move_limit: int = 20
) -> Dict[str, Any]:
    compact = deepcopy(full_payload)
    diagnosis = DiagnosePhase2Seed(payload=compact)
    moves = EnumeratePhase2RepairMoves(payload=compact, limit=move_limit)
    compact["tool_context"] = {
        "diagnosis": diagnosis,
        "enumerated_moves": moves.get("moves") or [],
        "scoring_hints": {
            "baseline_score": diagnosis.get("score"),
            "hard_valid": diagnosis.get("hard_valid"),
            "priority_clusters": diagnosis.get("prioritized_clusters") or [],
            "priority_objects": diagnosis.get("prioritized_objects") or [],
        },
    }
    return compact


# -----------------------------------------------------------------------------
# JSON extraction / parse
# -----------------------------------------------------------------------------
def _extract_json_like(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_llm_output(raw: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    blob = _extract_json_like(str(raw))
    return json.loads(blob)


# -----------------------------------------------------------------------------
# Proposal normalization / validation
# -----------------------------------------------------------------------------
def _normalize_object_repair_params(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(params or {})

    def _first(*keys: str) -> Any:
        for key in keys:
            if key in params and params.get(key) is not None:
                return params.get(key)
        return None

    def _parse_int(value: Any, field_name: str) -> int:
        if value is None:
            raise ValueError(f"{field_name} is required")
        if isinstance(value, str):
            value = value.strip().lower().replace("deg", "").strip()
        try:
            return int(float(value))
        except Exception as e:
            raise ValueError(f"{field_name} must be numeric") from e

    def _parse_float(value: Any, field_name: str) -> float:
        if value is None:
            raise ValueError(f"{field_name} is required")
        try:
            return float(value)
        except Exception as e:
            raise ValueError(f"{field_name} must be numeric") from e

    if op == "rotate_object":
        rot_raw = _first("rot", "rotation", "angle", "degrees")
        rot = _parse_int(rot_raw, "rotate_object.params.rot") % 360
        if rot not in {0, 90, 180, 270}:
            raise ValueError("rotate_object.params.rot must be one of 0/90/180/270")
        return {"rot": rot}

    if op == "mirror_object":
        axis = str(_first("axis", "mirror_axis", "direction") or "x").strip().lower()
        if axis not in {"x", "y"}:
            raise ValueError("mirror_object.params.axis must be 'x' or 'y'")
        return {"axis": axis}

    if op == "nudge_object":
        dx = _parse_int(_first("dx", "x", "offset_x"), "nudge_object.params.dx")
        dy = _parse_int(_first("dy", "y", "offset_y"), "nudge_object.params.dy")
        return {"dx": dx, "dy": dy}

    if op == "swap_objects":
        other = str(
            _first("other_object_id", "target_object_id", "swap_with", "other") or ""
        ).strip()
        if not other:
            raise ValueError("swap_objects.params.other_object_id is required")
        return {"other_object_id": other}

    if op == "set_anchor":
        anchor = str(_first("anchor", "anchor_id", "anchor_name", "side") or "").strip()
        if not anchor:
            raise ValueError("set_anchor.params.anchor is required")
        return {"anchor": anchor}

    if op == "set_front_override":
        fx = _first("dx", "fx", "front_dx", "x")
        fy = _first("dy", "fy", "front_dy", "y")
        return {
            "dx": _parse_float(fx, "set_front_override.params.dx"),
            "dy": _parse_float(fy, "set_front_override.params.dy"),
        }

    return params


def normalize_and_validate_repair_output(
    payload: Dict[str, Any], raw_output: str | Dict[str, Any]
) -> Dict[str, Any]:
    parsed = parse_llm_output(raw_output)
    try:
        proposal = Phase2RepairOutput.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"LLM repair output schema invalid: {e}") from e

    seed_layout = payload.get("seed_layout") or {}
    seed_tmap = _seed_transform_map(seed_layout)
    seed_vmap = _seed_variant_map(seed_layout)
    seed_cluster_ids = set(seed_tmap.keys())

    out_tmap = {**seed_tmap}
    for item in proposal.cluster_transforms:
        out_tmap[item.cluster_id] = item.model_dump()

    out_vmap = {**seed_vmap}
    for item in proposal.selected_variants:
        out_vmap[item.cluster_id] = item.model_dump()

    if set(out_tmap.keys()) != seed_cluster_ids:
        missing = sorted(seed_cluster_ids - set(out_tmap.keys()))
        extra = sorted(set(out_tmap.keys()) - seed_cluster_ids)
        raise ValueError(
            f"cluster_transforms cluster-id set mismatch. missing={missing}, extra={extra}"
        )

    if set(out_vmap.keys()) != seed_cluster_ids:
        missing = sorted(seed_cluster_ids - set(out_vmap.keys()))
        extra = sorted(set(out_vmap.keys()) - seed_cluster_ids)
        raise ValueError(
            f"selected_variants cluster-id set mismatch. missing={missing}, extra={extra}"
        )

    allowed_rots = _allowed_cluster_rotations(payload)
    for cid, item in out_tmap.items():
        rot = int(item.get("rot") or 0)
        if cid in allowed_rots and rot not in allowed_rots[cid]:
            raise ValueError(f"cluster {cid} rot={rot} not allowed by cluster rules")

    allowed_variants = _allowed_variant_ids_by_cluster(payload)
    for cid, item in out_vmap.items():
        vid = str(item.get("variant_id") or "")
        if cid in allowed_variants and vid not in allowed_variants[cid]:
            raise ValueError(f"cluster {cid} variant_id={vid} not allowed")

    allowed_obj_ids = _allowed_object_ids_by_cluster(payload)
    allowed_ops = _allowed_object_ops(payload)

    normalized_object_repairs: List[Dict[str, Any]] = []
    seen_object_repairs: set[tuple] = set()

    for item in proposal.object_repairs:
        row = item.model_dump()
        cid = row["cluster_id"]
        oid = row["object_id"]
        op = row["op"]

        if cid not in seed_cluster_ids:
            raise ValueError(f"object_repair references unknown cluster_id={cid}")
        cluster_obj_ids = allowed_obj_ids.get(cid, set())
        if oid not in cluster_obj_ids:
            raise ValueError(
                f"object_repair references unknown object_id={oid} in cluster={cid}"
            )
        if op not in allowed_ops:
            raise ValueError(f"object_repair op={op} not allowed by edit_contract")

        row["params"] = _normalize_object_repair_params(op, row.get("params") or {})

        if op == "swap_objects":
            other = str(row["params"]["other_object_id"]).strip()
            if other == oid:
                raise ValueError(
                    f"swap_objects cannot swap object_id={oid} with itself in cluster={cid}"
                )
            if other not in cluster_obj_ids:
                for other_cid, obj_ids in allowed_obj_ids.items():
                    if other in obj_ids and other_cid != cid:
                        raise ValueError(
                            f"swap_objects cross-cluster swap is not allowed: {cid}.{oid} cannot swap with {other_cid}.{other}"
                        )
                raise ValueError(
                    f"swap_objects.params.other_object_id={other} not found in cluster={cid}"
                )

        repair_sig = (
            cid,
            oid,
            op,
            json.dumps(row["params"], sort_keys=True, ensure_ascii=True),
        )
        if repair_sig in seen_object_repairs:
            continue
        seen_object_repairs.add(repair_sig)
        normalized_object_repairs.append(row)

    notes: List[str] = []
    seen_notes = set()
    for note in proposal.notes:
        if note not in seen_notes:
            seen_notes.add(note)
            notes.append(note)

    return {
        "status": proposal.status,
        "cluster_transforms": [out_tmap[cid] for cid in sorted(out_tmap.keys())],
        "selected_variants": [out_vmap[cid] for cid in sorted(out_vmap.keys())],
        "object_repairs": normalized_object_repairs,
        "notes": notes,
    }


# -----------------------------------------------------------------------------
# Compile using the same scorer/preview state
# -----------------------------------------------------------------------------
def compile_phase2_to_final_output(full_payload: Dict[str, Any]) -> Dict[str, Any]:
    phase2 = full_payload.get("phase2_placer") or {}
    repair = phase2.get("phase2_repair")
    if not isinstance(repair, dict):
        repair = {
            "status": "NO_IMPROVEMENT",
            "cluster_transforms": deepcopy(
                (full_payload.get("seed_layout") or {}).get("cluster_transforms") or []
            ),
            "selected_variants": deepcopy(
                (full_payload.get("seed_layout") or {}).get("selected_variants") or []
            ),
            "object_repairs": [],
            "notes": [],
        }
    preview = PreviewPhase2Repair(payload=full_payload, repair=repair)
    compiled = preview.get("compiled") or {}
    if not isinstance(compiled, dict):
        raise ValueError("PreviewPhase2Repair compiled output missing")
    _ensure_room_type(compiled, full_payload)
    return compiled


def _ensure_room_type(compiled: Dict[str, Any], full_payload: Dict[str, Any]) -> None:
    room = compiled.get("room")
    if not isinstance(room, dict):
        return
    if room.get("room_type"):
        return

    room_model = (full_payload.get("room_context") or {}).get("room_model_used") or {}
    meta = room_model.get("meta") if isinstance(room_model, dict) else None
    user_input = room_model.get("user_input") if isinstance(room_model, dict) else None

    room_type = None
    if isinstance(meta, dict):
        room_type = meta.get("room_type")
    if not room_type and isinstance(user_input, dict):
        room_type = user_input.get("room_type")

    if isinstance(room_type, str) and room_type:
        room["room_type"] = room_type
    else:
        room["room_type"] = "unknown"


# -----------------------------------------------------------------------------
# LLM runtime
# -----------------------------------------------------------------------------
def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _call_openai(*, system_prompt: str, user_payload_json: str) -> str | Dict[str, Any]:
    if OpenAIClient is None:
        raise RuntimeError("OpenAIClient is unavailable")
    client = OpenAIClient()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload_json},
    ]
    response = client.chat_completion(messages, model_key="gpt_4o", temperature=0.0)
    message = _extract_message(response)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("OpenAI response content missing")
    return content


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def _default_no_improvement(
    payload: Dict[str, Any], note: str | None = None
) -> Dict[str, Any]:
    seed = payload.get("seed_layout") or {}
    out = {
        "status": "NO_IMPROVEMENT",
        "cluster_transforms": deepcopy(seed.get("cluster_transforms") or []),
        "selected_variants": deepcopy(seed.get("selected_variants") or []),
        "object_repairs": [],
        "notes": [],
    }
    if note:
        out["notes"].append(str(note))
    return out


def run_phase2_placer(
    *,
    full_payload: Dict[str, Any],
    llm_call: LLMCallable,
    system_prompt: str = DEFAULT_PHASE2_PLACER_PROMPT,
    attach_result_to_payload: bool = True,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    compact_payload = prepare_phase2_llm_payload(full_payload)
    diagnosis = (compact_payload.get("tool_context") or {}).get("diagnosis") or {}
    baseline_score = int(diagnosis.get("score") or 0)
    best_repair = _default_no_improvement(compact_payload)
    best_scored = ScorePhase2Repair(payload=compact_payload, repair=best_repair)
    best_score = int(
        (best_scored.get("candidate") or {}).get("score") or baseline_score
    )

    attempt_errors: List[str] = []
    for attempt in range(max(1, int(max_attempts))):
        try:
            raw = llm_call(
                system_prompt=system_prompt,
                user_payload_json=json.dumps(
                    compact_payload, ensure_ascii=False, indent=2
                ),
            )
            normalized = normalize_and_validate_repair_output(compact_payload, raw)
            scored = ScorePhase2Repair(payload=compact_payload, repair=normalized)
            cand = scored.get("candidate") or {}
            cand_score = int(cand.get("score") or -(10**9))
            cand_valid = bool(cand.get("hard_valid"))
            if cand_valid and cand_score > best_score:
                best_repair = normalized
                best_scored = scored
                best_score = cand_score
        except Exception as e:
            attempt_errors.append(str(e))

    attached = deepcopy(full_payload)
    attached["phase2_placer"] = {
        "phase2_input": compact_payload,
        "phase2_repair": best_repair,
        "phase2_score": best_scored,
        "attempt_errors": attempt_errors,
    }
    attached["phase2_compiled_output"] = compile_phase2_to_final_output(attached)
    return attached if attach_result_to_payload else attached["phase2_placer"]


@dataclass(frozen=True)
class MacroClusterPlacer:
    system_prompt: str = DEFAULT_PHASE2_PLACER_PROMPT
    llm_call: LLMCallable = _call_openai

    def generate_raw(
        self,
        *,
        room_interpreter_json: Dict[str, Any],
        cluster_merged_json: Dict[str, Any],
        cluster_outlines_json: Dict[str, Any],
        relation_plan_json: Dict[str, Any],
        placer_output_json: Dict[str, Any],
        cluster_constraints_json: Dict[str, Any] | None = None,
    ) -> str:
        payload = build_phase2_payload(
            room_01=room_interpreter_json,
            clusters_04=cluster_merged_json,
            outlines_05=cluster_outlines_json,
            relation_05b=relation_plan_json,
            placer_06=placer_output_json,
            cluster_constraints=cluster_constraints_json,
        )
        result = run_phase2_placer(
            full_payload=payload,
            llm_call=self.llm_call,
            system_prompt=self.system_prompt,
            attach_result_to_payload=True,
        )
        compiled = result.get("phase2_compiled_output")
        if not isinstance(compiled, dict):
            raise ValueError("Phase-2 compiled output missing or invalid")
        return json.dumps(compiled, ensure_ascii=True)

    def generate(
        self,
        *,
        room_interpreter_json: Dict[str, Any],
        cluster_merged_json: Dict[str, Any],
        cluster_outlines_json: Dict[str, Any],
        relation_plan_json: Dict[str, Any],
        placer_output_json: Dict[str, Any],
        cluster_constraints_json: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raw = self.generate_raw(
            room_interpreter_json=room_interpreter_json,
            cluster_merged_json=cluster_merged_json,
            cluster_outlines_json=cluster_outlines_json,
            relation_plan_json=relation_plan_json,
            placer_output_json=placer_output_json,
            cluster_constraints_json=cluster_constraints_json,
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("MacroClusterPlacer response must be a JSON object")
        return payload


# -----------------------------------------------------------------------------
# Testing adapter
# -----------------------------------------------------------------------------
def make_file_response_adapter(path: str | Path) -> LLMCallable:
    def _call(*, system_prompt: str, user_payload_json: str) -> str | Dict[str, Any]:
        _ = system_prompt
        _ = user_payload_json
        return load_json(path)

    return _call


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase-2 Placer v3 with a prebuilt payload and a file-based mock LLM response."
    )
    parser.add_argument("--payload", required=True, help="Path to phase-2 payload JSON")
    parser.add_argument(
        "--mock-response", required=True, help="Path to mock LLM repair JSON"
    )
    parser.add_argument("--out", required=True, help="Path to write merged output JSON")
    args = parser.parse_args()

    payload = load_json(args.payload)
    adapter = make_file_response_adapter(args.mock_response)
    out = run_phase2_placer(full_payload=payload, llm_call=adapter)
    dump_json(args.out, out)
