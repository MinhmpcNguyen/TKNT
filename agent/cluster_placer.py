from __future__ import annotations

import importlib.util
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

try:
    from agent_schema.cluster_placer_schema import (
        MacroClusterPlacerOutput,  # type: ignore
    )
except Exception:
    MacroClusterPlacerOutput = None  # type: ignore


def _load_local_module(module_name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local module: {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from cluster_placer.tools import (  # type: ignore
        BuildGenericClusterVariants,
        EnumerateClusterCandidates,
        GlobalClusterVerifier,
        MaterializeVariantizedClusters,
        RankClusterPlacementOrder,
    )
except Exception:
    _tools_mod = _load_local_module(
        "tools_converge_fast_local", "tools_converge_fast.py"
    )
    GlobalClusterVerifier = _tools_mod.GlobalClusterVerifier
    EnumerateClusterCandidates = _tools_mod.EnumerateClusterCandidates
    RankClusterPlacementOrder = _tools_mod.RankClusterPlacementOrder
    BuildGenericClusterVariants = _tools_mod.BuildGenericClusterVariants
    MaterializeVariantizedClusters = _tools_mod.MaterializeVariantizedClusters


@dataclass(frozen=True)
class SearchConfig:
    grid_mm: int = 50
    max_variants_per_cluster: int = 6
    max_candidates_per_cluster: int = 48
    max_search_candidates_per_cluster: int = 16
    exact_search_combo_limit: int = 3000
    beam_width: int = 16
    beam_branch_per_state: int = 8
    local_repair_iterations: int = 16
    local_repair_branch: int = 8
    acceptable_critical_orientation_threshold_mm: int = 260
    acceptable_focal_pair_threshold_mm: int = 260
    acceptable_max_item_penalty_mm: int = 220


@dataclass
class SearchState:
    transforms_by_id: dict[str, dict[str, int]] = field(default_factory=dict)
    variants_by_id: dict[str, str] = field(default_factory=dict)
    score_tuple: tuple = field(default_factory=tuple)
    quality: dict[str, Any] = field(default_factory=dict)
    quality_gate: dict[str, Any] = field(default_factory=dict)
    repair_guidance: dict[str, Any] = field(default_factory=dict)
    state_signature: str = ""
    notes: list[str] = field(default_factory=list)

    def transforms_list(self) -> list[dict[str, int | str]]:
        out = []
        for cid in sorted(self.transforms_by_id):
            t = self.transforms_by_id[cid]
            out.append(
                {
                    "cluster_id": cid,
                    "x": int(t["x"]),
                    "y": int(t["y"]),
                    "rot": int(t["rot"]) % 360,
                }
            )
        return out

    def selected_variants_list(self) -> list[dict[str, str]]:
        out = []
        for cid in sorted(self.variants_by_id):
            vid = self.variants_by_id[cid]
            if isinstance(vid, str) and vid:
                out.append({"cluster_id": cid, "variant_id": vid})
        return out


@dataclass(frozen=True)
class MacroClusterPlacer:
    config: SearchConfig = SearchConfig()
    system_prompt: str = ""
    prompt_template: str = ""

    def generate_raw(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_outlines_json: dict[str, Any] | list[Any],
        relation_plan_json: dict[str, Any] | None = None,
        existing_transforms_json: dict[str, Any] | None = None,
        verifier_feedback_json: dict[str, Any] | None = None,
        grid_mm: int = 50,
    ) -> str:
        del verifier_feedback_json
        (
            clusters_outlines_json,
            filtered_relation_plan,
            existing_transforms_json,
        ) = _filter_unsat_input_clusters(
            clusters_outlines=clusters_outlines_json,
            relation_plan=relation_plan_json,
            existing_transforms=existing_transforms_json,
        )

        cfg = SearchConfig(
            grid_mm=int(grid_mm or self.config.grid_mm),
            max_variants_per_cluster=self.config.max_variants_per_cluster,
            max_candidates_per_cluster=self.config.max_candidates_per_cluster,
            max_search_candidates_per_cluster=self.config.max_search_candidates_per_cluster,
            exact_search_combo_limit=self.config.exact_search_combo_limit,
            beam_width=self.config.beam_width,
            beam_branch_per_state=self.config.beam_branch_per_state,
            local_repair_iterations=self.config.local_repair_iterations,
            local_repair_branch=self.config.local_repair_branch,
            acceptable_critical_orientation_threshold_mm=self.config.acceptable_critical_orientation_threshold_mm,
            acceptable_focal_pair_threshold_mm=self.config.acceptable_focal_pair_threshold_mm,
            acceptable_max_item_penalty_mm=self.config.acceptable_max_item_penalty_mm,
        )

        normalized_clusters, applied_offsets = _normalize_clusters_outlines_for_solver(
            clusters_outlines_json
        )
        normalized_existing = _normalize_existing_transforms_for_canonical_origins(
            existing_transforms_json,
            applied_offsets,
        )

        payload = _solve_layout(
            room_model=room_model_json,
            clusters_outlines=normalized_clusters,
            relation_plan=filtered_relation_plan,
            existing_transforms_json=normalized_existing,
            cfg=cfg,
        )
        payload = _denormalize_payload_from_canonical_origins(
            payload, normalized_clusters
        )
        return json.dumps(payload, ensure_ascii=True)

    def generate(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_outlines_json: dict[str, Any] | list[Any],
        relation_plan_json: dict[str, Any] | None = None,
        existing_transforms_json: dict[str, Any] | None = None,
        verifier_feedback_json: dict[str, Any] | None = None,
        grid_mm: int = 50,
    ) -> dict[str, Any]:
        raw = self.generate_raw(
            room_model_json=room_model_json,
            clusters_outlines_json=clusters_outlines_json,
            relation_plan_json=relation_plan_json,
            existing_transforms_json=existing_transforms_json,
            verifier_feedback_json=verifier_feedback_json,
            grid_mm=grid_mm,
        )
        payload = _parse_json(raw)
        payload = _sanitize_output_payload(payload)
        if MacroClusterPlacerOutput is not None:
            try:
                return MacroClusterPlacerOutput.model_validate(payload).model_dump()  # type: ignore[attr-defined]
            except Exception:
                logger.warning(
                    "MacroClusterPlacerOutput schema validation failed; returning raw payload with selected_variants preserved."
                )
        return payload


def _solve_layout(
    *,
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    existing_transforms_json: dict[str, Any] | None,
    cfg: SearchConfig,
) -> dict[str, Any]:
    cluster_ids = _cluster_ids(clusters_outlines)
    materialization_cache: dict[str, Any] = {}
    if not cluster_ids:
        return _build_need_info_json(
            grid_mm=cfg.grid_mm,
            notes=["No valid cluster entries were found in clusters_outlines_json."],
        )

    seed_state, seed_notes = _build_seed_state(
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan,
        existing_transforms_json=existing_transforms_json,
        cfg=cfg,
    )

    if len(seed_state.transforms_by_id) == len(cluster_ids):
        final_verify = _verify_layout(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            transforms=seed_state.transforms_list(),
            selected_variants=seed_state.variants_by_id,
            cfg=cfg,
            mode="complete",
            materialization_cache=materialization_cache,
        )
        if bool(final_verify.get("acceptable_valid")):
            return _build_ok_json(
                grid_mm=cfg.grid_mm,
                transforms=seed_state.transforms_list(),
                selected_variants=seed_state.selected_variants_list(),
                notes=[
                    *seed_notes,
                    "Existing transforms already formed an acceptable valid layout.",
                ],
            )

    enum = EnumerateClusterCandidates(
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        grid_mm=cfg.grid_mm,
        relation_plan=relation_plan,
        max_candidates_per_cluster=cfg.max_candidates_per_cluster,
        max_variants_per_cluster=cfg.max_variants_per_cluster,
        acceptable_critical_orientation_threshold_mm=cfg.acceptable_critical_orientation_threshold_mm,
        acceptable_focal_pair_threshold_mm=cfg.acceptable_focal_pair_threshold_mm,
        acceptable_max_item_penalty_mm=cfg.acceptable_max_item_penalty_mm,
    )
    if enum.get("result") != "OK":
        notes = [*seed_notes, "Candidate enumeration failed."]
        for err in enum.get("errors") or []:
            if isinstance(err, dict):
                notes.append(f"{err.get('code')}: {err.get('detail')}")
        return _build_unsat_json(grid_mm=cfg.grid_mm, notes=_dedupe_str_list(notes))

    order_result = RankClusterPlacementOrder(
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        grid_mm=cfg.grid_mm,
        relation_plan=relation_plan,
        max_variants_per_cluster=cfg.max_variants_per_cluster,
    )
    order = _derive_cluster_order(
        order_result, enum, cluster_ids, seed_state.transforms_by_id
    )
    candidate_map = _prepare_candidate_map(
        enum=enum,
        existing_transforms=seed_state.transforms_by_id,
        max_search_candidates_per_cluster=cfg.max_search_candidates_per_cluster,
    )

    missing_candidate_ids = [
        cid
        for cid in order
        if cid not in seed_state.transforms_by_id and not candidate_map.get(cid)
    ]
    if missing_candidate_ids:
        return _build_unsat_json(
            grid_mm=cfg.grid_mm,
            notes=_dedupe_str_list(
                [
                    *seed_notes,
                    f"No hard-valid candidates were found for clusters: {', '.join(missing_candidate_ids)}.",
                ]
            ),
        )

    remaining_ids = [cid for cid in order if cid not in seed_state.transforms_by_id]
    combo_count = 1
    for cid in remaining_ids:
        combo_count *= max(1, len(candidate_map.get(cid, [])))

    logger.info(
        "Deterministic MacroClusterPlacer: %s clusters, %s remaining after seed, estimated candidate product=%s",
        len(cluster_ids),
        len(remaining_ids),
        combo_count,
    )

    best_acceptable: SearchState | None = None
    best_hard: SearchState | None = None

    if not remaining_ids:
        verify = _verify_layout(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            transforms=seed_state.transforms_list(),
            selected_variants=seed_state.variants_by_id,
            cfg=cfg,
            mode="complete",
            materialization_cache=materialization_cache,
        )
        candidate = _state_from_verify(
            seed_state.transforms_by_id, seed_state.variants_by_id, verify
        )
        if bool(verify.get("acceptable_valid")):
            best_acceptable = candidate
        elif bool(verify.get("hard_valid")):
            best_hard = candidate
    elif combo_count <= cfg.exact_search_combo_limit:
        best_acceptable, best_hard = _exact_search(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            order=remaining_ids,
            candidate_map=candidate_map,
            seed_state=seed_state,
            cfg=cfg,
            materialization_cache=materialization_cache,
        )
    else:
        best_acceptable, best_hard = _beam_search(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            order=remaining_ids,
            candidate_map=candidate_map,
            seed_state=seed_state,
            cfg=cfg,
            materialization_cache=materialization_cache,
        )

    if best_acceptable is None and best_hard is not None:
        repaired = _conflict_directed_repair(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            candidate_map=candidate_map,
            seed_state=best_hard,
            cfg=cfg,
            materialization_cache=materialization_cache,
        )
        if repaired is not None and bool(repaired.quality_gate.get("pass")):
            best_acceptable = repaired
        elif repaired is not None and _better_state(repaired, best_hard):
            best_hard = repaired

    if best_acceptable is not None:
        return _build_ok_json(
            grid_mm=cfg.grid_mm,
            transforms=best_acceptable.transforms_list(),
            selected_variants=best_acceptable.selected_variants_list(),
            notes=_dedupe_str_list(
                [
                    *seed_notes,
                    "Solved with deterministic candidate+variant search.",
                    f"state_signature={best_acceptable.state_signature}",
                ]
            ),
        )

    if best_hard is not None:
        gate = best_hard.quality_gate or {}
        reasons = gate.get("reasons") or []
        return _build_unsat_json(
            grid_mm=cfg.grid_mm,
            notes=_dedupe_str_list(
                [
                    *seed_notes,
                    "Only hard-valid layouts were found; none passed the acceptable quality gate.",
                    f"quality_gate_reasons={json.dumps(reasons, ensure_ascii=True)}",
                    f"state_signature={best_hard.state_signature}",
                ]
            ),
        )

    return _build_unsat_json(
        grid_mm=cfg.grid_mm,
        notes=_dedupe_str_list(
            [
                *seed_notes,
                "Search terminated with no complete hard-valid layout.",
            ]
        ),
    )


def _exact_search(
    *,
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    order: list[str],
    candidate_map: dict[str, list[dict[str, Any]]],
    seed_state: SearchState,
    cfg: SearchConfig,
    materialization_cache: dict[str, Any],
) -> tuple[SearchState | None, SearchState | None]:
    best_acceptable: SearchState | None = None
    best_hard: SearchState | None = None
    visited: set[str] = set()

    def dfs(idx: int, state: SearchState) -> None:
        nonlocal best_acceptable, best_hard
        if idx >= len(order):
            verify = _verify_layout(
                room_model=room_model,
                clusters_outlines=clusters_outlines,
                relation_plan=relation_plan,
                transforms=state.transforms_list(),
                selected_variants=state.variants_by_id,
                cfg=cfg,
                mode="complete",
                materialization_cache=materialization_cache,
            )
            candidate = _state_from_verify(
                state.transforms_by_id, state.variants_by_id, verify
            )
            if bool(verify.get("acceptable_valid")):
                if _better_state(candidate, best_acceptable):
                    best_acceptable = candidate
            elif bool(verify.get("hard_valid")):
                if _better_state(candidate, best_hard):
                    best_hard = candidate
            return

        cid = order[idx]
        for cand in candidate_map.get(cid, []):
            next_transforms = dict(state.transforms_by_id)
            next_transforms[cid] = {
                "x": int(cand["x"]),
                "y": int(cand["y"]),
                "rot": int(cand["rot"]),
            }
            next_variants = dict(state.variants_by_id)
            next_variants[cid] = str(cand.get("variant_id") or _base_variant_id(cid))
            signature_seed = _state_key(next_transforms, next_variants)
            if signature_seed in visited:
                continue
            verify = _verify_layout(
                room_model=room_model,
                clusters_outlines=clusters_outlines,
                relation_plan=relation_plan,
                transforms=_transforms_dict_to_list(next_transforms),
                selected_variants=next_variants,
                cfg=cfg,
                mode="partial",
                materialization_cache=materialization_cache,
            )
            if not bool(verify.get("hard_valid")):
                continue
            next_state = _state_from_verify(next_transforms, next_variants, verify)
            sig = next_state.state_signature or signature_seed
            if sig in visited:
                continue
            visited.add(sig)
            dfs(idx + 1, next_state)

    dfs(0, seed_state)
    return best_acceptable, best_hard


def _beam_search(
    *,
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    order: list[str],
    candidate_map: dict[str, list[dict[str, Any]]],
    seed_state: SearchState,
    cfg: SearchConfig,
    materialization_cache: dict[str, Any],
) -> tuple[SearchState | None, SearchState | None]:
    beam: list[SearchState] = [seed_state]
    best_acceptable: SearchState | None = None
    best_hard: SearchState | None = None

    for cid in order:
        expanded: dict[str, SearchState] = {}
        for state in beam:
            taken = 0
            for cand in candidate_map.get(cid, []):
                next_transforms = dict(state.transforms_by_id)
                next_transforms[cid] = {
                    "x": int(cand["x"]),
                    "y": int(cand["y"]),
                    "rot": int(cand["rot"]),
                }
                next_variants = dict(state.variants_by_id)
                next_variants[cid] = str(
                    cand.get("variant_id") or _base_variant_id(cid)
                )
                verify = _verify_layout(
                    room_model=room_model,
                    clusters_outlines=clusters_outlines,
                    relation_plan=relation_plan,
                    transforms=_transforms_dict_to_list(next_transforms),
                    selected_variants=next_variants,
                    cfg=cfg,
                    mode="partial",
                    materialization_cache=materialization_cache,
                )
                if not bool(verify.get("hard_valid")):
                    continue
                next_state = _state_from_verify(next_transforms, next_variants, verify)
                sig = next_state.state_signature or _state_key(
                    next_transforms, next_variants
                )
                prev = expanded.get(sig)
                if _better_state(next_state, prev):
                    expanded[sig] = next_state
                taken += 1
                if taken >= cfg.beam_branch_per_state:
                    break
        if not expanded:
            return best_acceptable, best_hard
        beam = sorted(expanded.values(), key=_state_sort_key)[: cfg.beam_width]

    for state in beam:
        verify = _verify_layout(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            transforms=state.transforms_list(),
            selected_variants=state.variants_by_id,
            cfg=cfg,
            mode="complete",
            materialization_cache=materialization_cache,
        )
        done = _state_from_verify(state.transforms_by_id, state.variants_by_id, verify)
        if bool(verify.get("acceptable_valid")):
            if _better_state(done, best_acceptable):
                best_acceptable = done
        elif bool(verify.get("hard_valid")):
            if _better_state(done, best_hard):
                best_hard = done

    return best_acceptable, best_hard


def _conflict_directed_repair(
    *,
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    candidate_map: dict[str, list[dict[str, Any]]],
    seed_state: SearchState,
    cfg: SearchConfig,
    materialization_cache: dict[str, Any],
) -> SearchState | None:
    current = seed_state
    seen: set[str] = {current.state_signature or _state_key(current.transforms_by_id)}

    for _ in range(cfg.local_repair_iterations):
        prioritized = (current.repair_guidance or {}).get("prioritized_clusters") or []
        target_ids: list[str] = []
        for rec in prioritized:
            if isinstance(rec, dict):
                cid = rec.get("cluster_id")
                if isinstance(cid, str) and cid:
                    target_ids.append(cid)
        if not target_ids:
            break

        improved: SearchState | None = None
        for target_cid in target_ids[:3]:
            base_transforms = dict(current.transforms_by_id)
            base_variants = dict(current.variants_by_id)
            old = base_transforms.pop(target_cid, None)
            base_variants.pop(target_cid, None)
            if old is None:
                continue
            tried = 0
            for cand in candidate_map.get(target_cid, []):
                replacement = {
                    "x": int(cand["x"]),
                    "y": int(cand["y"]),
                    "rot": int(cand["rot"]),
                }
                replacement_variant = str(
                    cand.get("variant_id") or _base_variant_id(target_cid)
                )
                if (
                    replacement == old
                    and replacement_variant == current.variants_by_id.get(target_cid)
                ):
                    continue
                next_transforms = dict(base_transforms)
                next_transforms[target_cid] = replacement
                next_variants = dict(base_variants)
                next_variants[target_cid] = replacement_variant
                sig0 = _state_key(next_transforms, next_variants)
                if sig0 in seen:
                    continue
                verify = _verify_layout(
                    room_model=room_model,
                    clusters_outlines=clusters_outlines,
                    relation_plan=relation_plan,
                    transforms=_transforms_dict_to_list(next_transforms),
                    selected_variants=next_variants,
                    cfg=cfg,
                    mode="complete",
                    materialization_cache=materialization_cache,
                )
                if not bool(verify.get("hard_valid")):
                    continue
                candidate = _state_from_verify(next_transforms, next_variants, verify)
                seen.add(candidate.state_signature or sig0)
                if _better_state(candidate, improved):
                    improved = candidate
                if bool(candidate.quality_gate.get("pass")):
                    return candidate
                tried += 1
                if tried >= cfg.local_repair_branch:
                    break
        if improved is None or not _better_state(improved, current):
            break
        current = improved
        if bool(current.quality_gate.get("pass")):
            return current
    return current


def _prepare_candidate_map(
    *,
    enum: dict[str, Any],
    existing_transforms: dict[str, dict[str, int]],
    max_search_candidates_per_cluster: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in enum.get("clusters") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        if cid in existing_transforms:
            out[cid] = []
            continue
        raw_candidates = [
            c
            for c in (item.get("candidates") or [])
            if isinstance(c, dict) and bool(c.get("hard_valid"))
        ]
        out[cid] = _stratify_candidates(
            raw_candidates, max_search_candidates_per_cluster
        )
    return out


def _stratify_candidates(
    candidates: list[dict[str, Any]],
    max_count: int,
) -> list[dict[str, Any]]:
    if len(candidates) <= max_count:
        return candidates

    def cand_key(c: dict[str, Any]) -> tuple:
        return (
            0 if c.get("acceptable_valid") else 1,
            -float(c.get("variant_priority") or 0.0),
            -int(c.get("anchor_priority") or 0),
            -int(c.get("rough_score") or -(10**9)),
            int(c.get("critical_orientation_penalty_mm") or 0),
            int(c.get("orientation_penalty_mm") or 0),
            str(c.get("variant_family") or ""),
            str(c.get("anchor_kind") or ""),
            int(c.get("rot") or 0),
        )

    candidates = sorted(candidates, key=cand_key)
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for cand in candidates:
        bucket = (
            str(cand.get("variant_family") or cand.get("variant_id") or ""),
            str(cand.get("anchor_kind") or ""),
            int(cand.get("rot") or 0) % 360,
        )
        if bucket in seen:
            continue
        seen.add(bucket)
        chosen.append(cand)
        if len(chosen) >= max_count:
            return chosen

    for cand in candidates:
        sig = (
            str(cand.get("cluster_id") or ""),
            int(cand.get("x") or 0),
            int(cand.get("y") or 0),
            int(cand.get("rot") or 0),
            str(cand.get("variant_id") or ""),
        )
        if any(
            sig
            == (
                str(x.get("cluster_id") or ""),
                int(x.get("x") or 0),
                int(x.get("y") or 0),
                int(x.get("rot") or 0),
                str(x.get("variant_id") or ""),
            )
            for x in chosen
        ):
            continue
        chosen.append(cand)
        if len(chosen) >= max_count:
            break
    return chosen


def _derive_cluster_order(
    order_result: dict[str, Any],
    enum: dict[str, Any],
    cluster_ids: list[str],
    locked: dict[str, dict[str, int]],
) -> list[str]:
    ranked: list[str] = []
    if isinstance(order_result, dict) and order_result.get("result") == "OK":
        for row in order_result.get("order") or []:
            if isinstance(row, dict):
                cid = row.get("cluster_id")
                if isinstance(cid, str) and cid in cluster_ids and cid not in ranked:
                    ranked.append(cid)
    for item in enum.get("clusters") or []:
        if isinstance(item, dict):
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid in cluster_ids and cid not in ranked:
                ranked.append(cid)
    for cid in sorted(cluster_ids):
        if cid not in ranked:
            ranked.append(cid)
    return [cid for cid in ranked if cid not in locked]


def _build_seed_state(
    *,
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    existing_transforms_json: dict[str, Any] | None,
    cfg: SearchConfig,
) -> tuple[SearchState, list[str]]:
    notes: list[str] = []
    seed_transforms: dict[str, dict[str, int]] = {}

    raw = []
    if isinstance(existing_transforms_json, dict):
        raw = existing_transforms_json.get("cluster_transforms") or []
    parsed = _canonicalize_cluster_transforms(raw)
    if not parsed:
        return SearchState(), notes

    for item in parsed:
        cid = item["cluster_id"]
        candidate_transforms = dict(seed_transforms)
        candidate_transforms[cid] = {
            "x": item["x"],
            "y": item["y"],
            "rot": item["rot"],
        }
        candidate_variants = {
            **{k: _base_variant_id(k) for k in candidate_transforms.keys()}
        }
        verify = _verify_layout(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            transforms=_transforms_dict_to_list(candidate_transforms),
            selected_variants=candidate_variants,
            cfg=cfg,
            mode="partial",
            materialization_cache={},
        )
        if bool(verify.get("hard_valid")):
            seed_transforms[cid] = candidate_transforms[cid]
        else:
            notes.append(f"Ignored invalid existing transform for cluster '{cid}'.")

    if seed_transforms:
        seed_variants = {cid: _base_variant_id(cid) for cid in seed_transforms.keys()}
        verify = _verify_layout(
            room_model=room_model,
            clusters_outlines=clusters_outlines,
            relation_plan=relation_plan,
            transforms=_transforms_dict_to_list(seed_transforms),
            selected_variants=seed_variants,
            cfg=cfg,
            mode="partial",
            materialization_cache={},
        )
        return _state_from_verify(
            seed_transforms, seed_variants, verify
        ), _dedupe_str_list(notes)

    return SearchState(), _dedupe_str_list(notes)


def _verify_layout(
    *,
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    transforms: list[dict[str, Any]],
    selected_variants: dict[str, str] | None,
    cfg: SearchConfig,
    mode: str,
    materialization_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materialized_clusters = _materialize_clusters_for_variants(
        clusters_outlines=clusters_outlines,
        selected_variants=selected_variants,
        materialization_cache=materialization_cache,
    )
    log_args = {
        "mode": mode,
        "grid_mm": cfg.grid_mm,
        "cluster_transforms_count": len(transforms),
        "selected_variants_count": len(selected_variants or {}),
        "cluster_ids": [t.get("cluster_id") for t in transforms if isinstance(t, dict)],
    }
    if logger.isEnabledFor(logging.INFO):
        logger.info("Tool calls requested: %s", ["GlobalClusterVerifier"])
        logger.info(
            "Tool GlobalClusterVerifier args: %s",
            _short_json(log_args),
        )
    result = GlobalClusterVerifier(
        room_model=room_model,
        clusters_outlines=materialized_clusters,
        cluster_transforms=transforms,
        grid_mm=cfg.grid_mm,
        mode=mode,
        relation_plan=relation_plan,
        acceptable_critical_orientation_threshold_mm=cfg.acceptable_critical_orientation_threshold_mm,
        acceptable_focal_pair_threshold_mm=cfg.acceptable_focal_pair_threshold_mm,
        acceptable_max_item_penalty_mm=cfg.acceptable_max_item_penalty_mm,
        return_debug=True,
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Tool GlobalClusterVerifier output: %s",
            _short_json(result),
        )
    return result


def _state_from_verify(
    transforms_by_id: dict[str, dict[str, int]],
    variants_by_id: dict[str, str],
    verify: dict[str, Any],
) -> SearchState:
    quality = verify.get("quality") or {}
    gate = verify.get("quality_gate") or {}
    metrics = gate.get("metrics") or {}
    layout_score = int(quality.get("layout_score") or -(10**9))
    pass_flag = bool(gate.get("pass"))
    hard_valid = bool(verify.get("hard_valid"))
    complete = bool(verify.get("complete"))
    reasons = tuple(sorted(str(r) for r in (gate.get("reasons") or [])))
    score_tuple = (
        0 if pass_flag else 1,
        0 if hard_valid else 1,
        0 if complete else 1,
        len(reasons),
        int(metrics.get("critical_orientation_penalty_mm") or 0),
        int(metrics.get("focal_pair_penalty_mm") or 0),
        int(metrics.get("max_critical_item_penalty_mm") or 0),
        -layout_score,
    )
    return SearchState(
        transforms_by_id={cid: dict(v) for cid, v in transforms_by_id.items()},
        variants_by_id={
            cid: str(v) for cid, v in variants_by_id.items() if isinstance(v, str) and v
        },
        score_tuple=score_tuple,
        quality=quality,
        quality_gate=gate,
        repair_guidance=verify.get("repair_guidance") or {},
        state_signature=str(
            verify.get("state_signature")
            or _state_key(transforms_by_id, variants_by_id)
        ),
        notes=[],
    )


def _state_sort_key(state: SearchState) -> tuple:
    return state.score_tuple


def _better_state(candidate: SearchState | None, incumbent: SearchState | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return _state_sort_key(candidate) < _state_sort_key(incumbent)


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError("MacroClusterPlacer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("MacroClusterPlacer response must be a JSON object")
    return payload


def _truncate_text(text: str, max_len: int = 1200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _short_json(obj: Any, max_len: int = 1200) -> str:
    try:
        dumped = json.dumps(obj, ensure_ascii=True)
    except Exception:
        dumped = str(obj)
    return _truncate_text(dumped, max_len=max_len)


def _sanitize_output_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status not in {"OK", "UNSAT", "NEED_INFO"}:
        status = "UNSAT"
    notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    transforms = _canonicalize_cluster_transforms(
        payload.get("cluster_transforms") or []
    )
    selected_variants = _canonicalize_selected_variants(
        payload.get("selected_variants") or []
    )
    try:
        grid_mm = int(payload.get("grid_mm") or 50)
    except Exception:
        grid_mm = 50
    return {
        "status": status,
        "grid_mm": grid_mm,
        "cluster_transforms": transforms,
        "selected_variants": _selected_variants_dict_to_list(selected_variants),
        "notes": _dedupe_str_list([n for n in notes if isinstance(n, str)]),
    }


def _filter_unsat_input_clusters(
    *,
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    existing_transforms: dict[str, Any] | None,
) -> tuple[dict[str, Any] | list[Any], dict[str, Any] | None, dict[str, Any] | None]:
    filtered_outlines, removed_ids = _filter_unsat_clusters_outlines(
        clusters_outlines
    )
    if not removed_ids:
        return clusters_outlines, relation_plan, existing_transforms
    filtered_relation_plan = _filter_relation_plan_clusters(relation_plan, removed_ids)
    filtered_existing = _filter_existing_transforms(existing_transforms, removed_ids)
    return filtered_outlines, filtered_relation_plan, filtered_existing


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
        out[key] = [item for item in items if isinstance(item, dict) and predicate(item)]

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


def _filter_existing_transforms(
    existing_transforms: dict[str, Any] | None, removed_ids: set[str]
) -> dict[str, Any] | None:
    if not removed_ids or not isinstance(existing_transforms, dict):
        return existing_transforms
    transforms = existing_transforms.get("cluster_transforms")
    if not isinstance(transforms, list):
        return existing_transforms
    kept = [
        item
        for item in transforms
        if not (
            isinstance(item, dict)
            and isinstance(item.get("cluster_id"), str)
            and item.get("cluster_id") in removed_ids
        )
    ]
    out = dict(existing_transforms)
    out["cluster_transforms"] = kept
    return out


def _build_ok_json(
    *,
    grid_mm: int,
    transforms: list[dict[str, Any]],
    selected_variants: list[dict[str, str]],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "status": "OK",
        "grid_mm": int(grid_mm),
        "cluster_transforms": _canonicalize_cluster_transforms(transforms),
        "selected_variants": _selected_variants_dict_to_list(
            _canonicalize_selected_variants(selected_variants)
        ),
        "notes": _dedupe_str_list(notes),
    }


def _build_unsat_json(*, grid_mm: int, notes: list[str]) -> dict[str, Any]:
    return {
        "status": "UNSAT",
        "grid_mm": int(grid_mm),
        "cluster_transforms": [],
        "selected_variants": [],
        "notes": _dedupe_str_list(notes),
    }


def _build_need_info_json(*, grid_mm: int, notes: list[str]) -> dict[str, Any]:
    return {
        "status": "NEED_INFO",
        "grid_mm": int(grid_mm),
        "cluster_transforms": [],
        "selected_variants": [],
        "notes": _dedupe_str_list(notes),
    }


def _canonicalize_cluster_transforms(raw: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            continue
        try:
            x = int(round(float(item.get("x", 0))))
            y = int(round(float(item.get("y", 0))))
            rot = int(item.get("rot", 0)) % 360
        except Exception:
            continue
        if rot not in {0, 90, 180, 270}:
            continue
        out.append({"cluster_id": cid, "x": x, "y": y, "rot": rot})
        seen.add(cid)
    out.sort(key=lambda item: item["cluster_id"])
    return out


def _state_key(
    transforms_by_id: dict[str, dict[str, int]],
    variants_by_id: dict[str, str] | None = None,
) -> str:
    payload = []
    variants_by_id = variants_by_id or {}
    for cid in sorted(transforms_by_id):
        t = transforms_by_id[cid]
        payload.append(
            (
                cid,
                int(t["x"]),
                int(t["y"]),
                int(t["rot"]) % 360,
                str(variants_by_id.get(cid) or _base_variant_id(cid)),
            )
        )
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _transforms_dict_to_list(
    transforms_by_id: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    out = []
    for cid in sorted(transforms_by_id):
        t = transforms_by_id[cid]
        out.append(
            {
                "cluster_id": cid,
                "x": int(t["x"]),
                "y": int(t["y"]),
                "rot": int(t["rot"]) % 360,
            }
        )
    return out


def _canonicalize_selected_variants(raw: Iterable[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for cid, vid in raw.items():
            if isinstance(cid, str) and cid and isinstance(vid, str) and vid:
                out[cid] = vid
        return out
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        vid = item.get("variant_id")
        if isinstance(cid, str) and cid and isinstance(vid, str) and vid:
            out[cid] = vid
    return out


def _selected_variants_dict_to_list(
    selected_variants: dict[str, str],
) -> list[dict[str, str]]:
    out = []
    for cid in sorted(selected_variants):
        vid = selected_variants[cid]
        if isinstance(vid, str) and vid:
            out.append({"cluster_id": cid, "variant_id": vid})
    return out


def _base_variant_id(cluster_id: str) -> str:
    return f"{cluster_id}::base"


def _materialize_clusters_for_variants(
    *,
    clusters_outlines: dict[str, Any] | list[Any],
    selected_variants: dict[str, str] | None,
    materialization_cache: dict[str, Any] | None,
) -> dict[str, Any] | list[Any]:
    selected_variants = {
        cid: vid
        for cid, vid in (selected_variants or {}).items()
        if isinstance(cid, str) and cid and isinstance(vid, str) and vid
    }
    # keep explicit base variants too so downstream signatures stay deterministic
    signature = json.dumps(
        sorted((cid, vid) for cid, vid in selected_variants.items()),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if materialization_cache is not None and signature in materialization_cache:
        return materialization_cache[signature]
    if not selected_variants:
        if materialization_cache is not None:
            materialization_cache[signature] = clusters_outlines
        return clusters_outlines
    try:
        result = MaterializeVariantizedClusters(
            clusters_outlines=clusters_outlines,
            selected_variants=selected_variants,
        )
        if (
            isinstance(result, dict)
            and result.get("result") == "OK"
            and result.get("clusters_outlines") is not None
        ):
            payload = result.get("clusters_outlines")
            if materialization_cache is not None:
                materialization_cache[signature] = payload
            return payload
    except Exception:
        logger.exception(
            "MaterializeVariantizedClusters failed; falling back to original clusters_outlines"
        )
    if materialization_cache is not None:
        materialization_cache[signature] = clusters_outlines
    return clusters_outlines


def _dedupe_str_list(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _cluster_ids(clusters_outlines: dict[str, Any] | list[Any]) -> list[str]:
    return [cid for cid, _ in _iter_cluster_entries_for_index(clusters_outlines)]


def _iter_cluster_entries_for_index(clusters_outlines: dict[str, Any] | list[Any]):
    if isinstance(clusters_outlines, dict):
        direct = []
        for cid, value in clusters_outlines.items():
            if (
                isinstance(cid, str)
                and isinstance(value, dict)
                and _looks_like_cluster_payload(value, cid)
            ):
                direct.append((cid, value))
        if direct:
            for item in sorted(direct, key=lambda x: x[0]):
                yield item
            return
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            for item in clusters:
                if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
                    yield (str(item["cluster_id"]), item)
            return
    elif isinstance(clusters_outlines, list):
        for item in clusters_outlines:
            if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
                yield (str(item["cluster_id"]), item)


def _looks_like_cluster_payload(
    value: dict[str, Any], fallback_cluster_id: str
) -> bool:
    cid = value.get("cluster_id")
    if isinstance(cid, str) and cid:
        return True
    if (
        value.get("cluster_footprint")
        or value.get("local_placements")
        or value.get("orientation_meta")
    ):
        return True
    return bool(fallback_cluster_id)


def _normalize_clusters_outlines_for_solver(
    clusters_outlines: dict[str, Any] | list[Any],
) -> tuple[dict[str, Any] | list[Any], dict[str, tuple[int, int]]]:
    applied_offsets: dict[str, tuple[int, int]] = {}

    if isinstance(clusters_outlines, dict):
        direct_out: dict[str, Any] = {}
        for cid, value in clusters_outlines.items():
            if not isinstance(cid, str) or not isinstance(value, dict):
                continue
            if not _looks_like_cluster_payload(value, cid):
                continue
            row = deepcopy(value)
            if not isinstance(row.get("cluster_id"), str) or not row.get("cluster_id"):
                row["cluster_id"] = cid
            row, applied = _canonicalize_cluster_payload_local_origin(row)
            direct_out[cid] = row
            applied_offsets[cid] = applied
        if direct_out:
            return direct_out, applied_offsets

        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            out_list: list[dict[str, Any]] = []
            for item in clusters:
                if not isinstance(item, dict):
                    continue
                cid = item.get("cluster_id")
                if not isinstance(cid, str) or not cid:
                    continue
                row, applied = _canonicalize_cluster_payload_local_origin(
                    deepcopy(item)
                )
                out_list.append(row)
                applied_offsets[cid] = applied
            if out_list:
                return out_list, applied_offsets

    if isinstance(clusters_outlines, list):
        out_list = []
        for item in clusters_outlines:
            if not isinstance(item, dict):
                continue
            cid = item.get("cluster_id")
            if not isinstance(cid, str) or not cid:
                continue
            row, applied = _canonicalize_cluster_payload_local_origin(deepcopy(item))
            out_list.append(row)
            applied_offsets[cid] = applied
        return out_list, applied_offsets

    return clusters_outlines, applied_offsets


def _canonicalize_cluster_payload_local_origin(
    cluster_payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int]]:
    row = deepcopy(cluster_payload)
    min_x, min_y = _infer_cluster_local_origin_offset(row)
    applied = (int(min_x), int(min_y))
    if min_x != 0 or min_y != 0:
        _shift_cluster_payload_local_geometry(row, dx=-min_x, dy=-min_y)
    prior_offset = _read_cluster_canonical_local_origin_offset(row)
    total_offset = (prior_offset[0] + min_x, prior_offset[1] + min_y)
    row["canonical_local_origin_offset"] = {
        "dx": int(total_offset[0]),
        "dy": int(total_offset[1]),
    }
    row["canonical_local_origin_normalized"] = True
    local_frame = row.get("local_frame")
    if isinstance(local_frame, dict):
        local_frame["origin_note"] = (
            "(0,0) is the canonical top-left of the occupied local bbox for this cluster"
        )
    return row, applied


def _infer_cluster_local_origin_offset(
    cluster_payload: dict[str, Any],
) -> tuple[int, int]:
    xs: list[float] = []
    ys: list[float] = []
    fp = cluster_payload.get("cluster_footprint")
    if isinstance(fp, dict):
        rects = fp.get("rects")
        if isinstance(rects, list):
            for r in rects:
                if not isinstance(r, dict):
                    continue
                try:
                    x = float(r.get("x", 0))
                    y = float(r.get("y", 0))
                    w = float(r.get("w", 0))
                    h = float(r.get("h", 0))
                except Exception:
                    continue
                xs.extend([x, x + max(w, 0.0)])
                ys.extend([y, y + max(h, 0.0)])
        outlines = fp.get("outline_polygons_ccw") or cluster_payload.get(
            "outline_polygons_ccw"
        )
        if isinstance(outlines, list):
            for poly in outlines:
                if not isinstance(poly, list):
                    continue
                for p in poly:
                    if not isinstance(p, dict):
                        continue
                    try:
                        xs.append(float(p.get("x", 0)))
                        ys.append(float(p.get("y", 0)))
                    except Exception:
                        continue
        bbox = fp.get("local_bbox")
        if isinstance(bbox, dict):
            try:
                xs.append(float(bbox.get("min_x", 0)))
                ys.append(float(bbox.get("min_y", 0)))
            except Exception:
                pass
    placements = cluster_payload.get("local_placements")
    if isinstance(placements, list):
        for p in placements:
            if not isinstance(p, dict):
                continue
            try:
                xs.append(float(p.get("x", 0)))
                ys.append(float(p.get("y", 0)))
            except Exception:
                continue
    if not xs or not ys:
        return (0, 0)
    return (int(round(min(xs))), int(round(min(ys))))


def _shift_cluster_payload_local_geometry(
    cluster_payload: dict[str, Any], *, dx: int, dy: int
) -> None:
    placements = cluster_payload.get("local_placements")
    if isinstance(placements, list):
        for p in placements:
            if not isinstance(p, dict):
                continue
            if isinstance(p.get("x"), (int, float)):
                p["x"] = int(round(float(p.get("x", 0)) + dx))
            if isinstance(p.get("y"), (int, float)):
                p["y"] = int(round(float(p.get("y", 0)) + dy))
    fp = cluster_payload.get("cluster_footprint")
    if not isinstance(fp, dict):
        return
    rects = fp.get("rects")
    if isinstance(rects, list):
        for r in rects:
            if not isinstance(r, dict):
                continue
            if isinstance(r.get("x"), (int, float)):
                r["x"] = int(round(float(r.get("x", 0)) + dx))
            if isinstance(r.get("y"), (int, float)):
                r["y"] = int(round(float(r.get("y", 0)) + dy))
    bbox = fp.get("local_bbox")
    if isinstance(bbox, dict):
        for key, delta in (("min_x", dx), ("max_x", dx), ("min_y", dy), ("max_y", dy)):
            if isinstance(bbox.get(key), (int, float)):
                bbox[key] = int(round(float(bbox.get(key, 0)) + delta))
    outlines = fp.get("outline_polygons_ccw")
    if isinstance(outlines, list):
        for poly in outlines:
            if not isinstance(poly, list):
                continue
            for p in poly:
                if not isinstance(p, dict):
                    continue
                if isinstance(p.get("x"), (int, float)):
                    p["x"] = int(round(float(p.get("x", 0)) + dx))
                if isinstance(p.get("y"), (int, float)):
                    p["y"] = int(round(float(p.get("y", 0)) + dy))


def _read_cluster_canonical_local_origin_offset(
    cluster_payload: dict[str, Any],
) -> tuple[int, int]:
    raw = cluster_payload.get("canonical_local_origin_offset")
    if isinstance(raw, dict):
        try:
            return (int(raw.get("dx", 0) or 0), int(raw.get("dy", 0) or 0))
        except Exception:
            return (0, 0)
    return (0, 0)


def _normalize_existing_transforms_for_canonical_origins(
    existing_transforms_json: dict[str, Any] | None,
    applied_offsets: dict[str, tuple[int, int]],
) -> dict[str, Any] | None:
    if not isinstance(existing_transforms_json, dict):
        return existing_transforms_json
    transforms = existing_transforms_json.get("cluster_transforms")
    if not isinstance(transforms, list):
        return existing_transforms_json
    out = dict(existing_transforms_json)
    out["cluster_transforms"] = _remap_cluster_transforms_by_local_origin(
        transforms,
        offsets=applied_offsets,
        to_canonical=True,
    )
    return out


def _denormalize_payload_from_canonical_origins(
    payload: dict[str, Any],
    clusters_outlines: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    transforms = out.get("cluster_transforms")
    if not isinstance(transforms, list):
        return out
    offsets = _collect_cluster_local_origin_offsets(clusters_outlines)
    out["cluster_transforms"] = _remap_cluster_transforms_by_local_origin(
        transforms,
        offsets=offsets,
        to_canonical=False,
    )
    return out


def _collect_cluster_local_origin_offsets(
    clusters_outlines: dict[str, Any] | list[Any],
) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    for cid, cinfo in _iter_cluster_entries_for_index(clusters_outlines):
        offsets[cid] = _read_cluster_canonical_local_origin_offset(cinfo)
    return offsets


def _remap_cluster_transforms_by_local_origin(
    transforms: list[dict[str, Any]],
    *,
    offsets: dict[str, tuple[int, int]],
    to_canonical: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in _canonicalize_cluster_transforms(transforms):
        cid = t["cluster_id"]
        off_x, off_y = offsets.get(cid, (0, 0))
        rdx, rdy = _rotate_local_offset_ccw_90s(off_x, off_y, t["rot"])
        sign = 1 if to_canonical else -1
        out.append(
            {
                "cluster_id": cid,
                "x": int(t["x"] + sign * rdx),
                "y": int(t["y"] + sign * rdy),
                "rot": int(t["rot"]) % 360,
            }
        )
    out.sort(key=lambda item: item["cluster_id"])
    return out


def _rotate_local_offset_ccw_90s(x: int, y: int, rot: int) -> tuple[int, int]:
    r = int(rot or 0) % 360
    if r == 0:
        return (int(x), int(y))
    if r == 90:
        return (-int(y), int(x))
    if r == 180:
        return (-int(x), -int(y))
    if r == 270:
        return (int(y), -int(x))
    return (int(x), int(y))


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Deterministic MacroClusterPlacer")
#     parser.add_argument("room_json")
#     parser.add_argument("clusters_json")
#     parser.add_argument("relation_json")
#     parser.add_argument("--existing", default=None)
#     parser.add_argument("--grid", type=int, default=50)
#     args = parser.parse_args()

#     room = json.loads(Path(args.room_json).read_text())
#     clusters = json.loads(Path(args.clusters_json).read_text())
#     relation = json.loads(Path(args.relation_json).read_text())
#     existing = json.loads(Path(args.existing).read_text()) if args.existing else None

#     placer = MacroClusterPlacer()
#     raw = placer.generate_raw(
#         room_model_json=room,
#         clusters_outlines_json=clusters,
#         relation_plan_json=relation,
#         existing_transforms_json=existing,
#         grid_mm=args.grid,
#     )
#     print(raw)
