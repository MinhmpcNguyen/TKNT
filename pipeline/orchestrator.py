from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.cluster_composer import ClusterComposer
from agent.cluster_forge import ClusterForge
from agent.cluster_relation_planner import ClusterRelationPlanner
from agent.phase2_controller import Phase2Controller
from agent.room_interpreter import RoomInterpreter
from agent.solver.solver import MacroClusterSolver
from agent.stylist import Stylist
from agent.tier_count_director import TierCountDirector
from cluster_composer.merge import merge_cluster_outputs
from cluster_composer.outline import compute_cluster_outline
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from pipeline.layout_variants import build_layout_variants_artifact
from pipeline.manual_placements import (
    apply_manual_placements_to_tier_output,
    append_manual_placements_guidance,
    augment_cluster_forge_with_manual_placements,
    merge_manual_placements_into_absolute_layout,
    merge_manual_placements_into_layout_variants,
    merge_manual_placements_into_styled_output,
    prepare_input_payload_for_manual_placements,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    root: Path

    @property
    def room_interpreter(self) -> Path:
        return self.root / "01_room_interpreter.json"

    @property
    def cluster_forge(self) -> Path:
        return self.root / "02_cluster_forge.json"

    @property
    def tier_count(self) -> Path:
        return self.root / "03_tier_count.json"

    @property
    def cluster_merged(self) -> Path:
        return self.root / "04_cluster_merged.json"

    @property
    def clusters_dir(self) -> Path:
        return self.root / "clusters"

    def cluster_composer(self, cluster_id: str) -> Path:
        return self.clusters_dir / f"cluster_composer_{cluster_id}.json"

    def cluster_outline(self, cluster_id: str) -> Path:
        return self.clusters_dir / f"cluster_outline_{cluster_id}.json"

    @property
    def cluster_outlines_all(self) -> Path:
        return self.root / "05_cluster_outlines.json"

    @property
    def cluster_relation_plan(self) -> Path:
        return self.root / "05b_cluster_relation_plan.json"

    @property
    def cluster_placer(self) -> Path:
        return self.root / "06_cluster_placer.json"

    @property
    def cluster_solver(self) -> Path:
        return self.root / "06a_cluster_solver.json"

    @property
    def absolute_layout(self) -> Path:
        return self.root / "07_absolute_layout.json"

    @property
    def stylist(self) -> Path:
        return self.root / "08_stylist.json"

    @property
    def layout_variants(self) -> Path:
        return self.root / "09_layout_variants.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def meta(self) -> Path:
        return self.root / "case_meta.json"


def _build_primary_guidance_text(
    *,
    room_output: dict[str, Any],
    input_payload: dict[str, Any],
    description: str | None,
    special_notes: str | None,
) -> str:
    notes = room_output.get("notes")
    if isinstance(notes, list):
        cleaned_notes = [
            note.strip() for note in notes if isinstance(note, str) and note.strip()
        ]
        if cleaned_notes:
            return "\n".join(f"- {note}" for note in cleaned_notes)

    legacy_parts: list[str] = []
    for value in (description, special_notes):
        if isinstance(value, str) and value.strip():
            legacy_parts.append(value.strip())

    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
            "style",
        ):
            value = user_input.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if text and text not in legacy_parts:
                legacy_parts.append(text)

    return "\n".join(legacy_parts)


def run_case(
    *,
    input_payload: dict[str, Any],
    user_id: str,
    description: str | None = None,
    special_notes: str | None = None,
    cases_root: str | Path = "cases",
    case_id: str | None = None,
) -> dict[str, Any]:
    prepared_input_payload, manual_placements = (
        prepare_input_payload_for_manual_placements(input_payload)
    )
    case_id = case_id or _make_case_id(user_id)
    case_root = Path(cases_root) / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    paths = CasePaths(case_id=case_id, root=case_root)
    paths.clusters_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.meta,
        {
            "case_id": case_id,
            "user_id": user_id,
            "created_at_utc": _now_utc_iso(),
        },
    )
    _update_status(paths, "room_interpreter")

    # 1) RoomInterpreter
    room_interpreter = RoomInterpreter()
    room_output = room_interpreter.generate(
        prepared_input_payload,
        description=description,
        special_notes=special_notes,
    ).model_dump()
    room_output = _strip_raw_text(room_output)
    _write_json(paths.room_interpreter, room_output)
    guidance_text = _build_primary_guidance_text(
        room_output=room_output,
        input_payload=prepared_input_payload,
        description=description,
        special_notes=special_notes,
    )
    planning_guidance_text = append_manual_placements_guidance(
        guidance_text,
        manual_placements,
    )

    _update_status(paths, "cluster_forge")

    # 2) ClusterForge
    room_type = _extract_room_type(prepared_input_payload, room_output)
    cluster_forge = ClusterForge()
    cluster_output = cluster_forge.generate(
        room_type,
        description=planning_guidance_text or None,
        special_notes=None,
    ).model_dump()
    cluster_output = _strip_raw_text(cluster_output)
    cluster_output = augment_cluster_forge_with_manual_placements(
        cluster_output,
        manual_placements,
    )
    _write_json(paths.cluster_forge, cluster_output)

    _update_status(paths, "tier_count")

    # 3) TierCountDirector
    tier_count = TierCountDirector()
    tier_output = tier_count.generate(
        description=planning_guidance_text,
        special_notes="",
        room_model_json=room_output,
        user_intent_json=prepared_input_payload,
        clusters_json=cluster_output,
    )
    tier_output = _strip_raw_text(tier_output)
    tier_output = apply_manual_placements_to_tier_output(
        tier_output,
        manual_placements,
        clusters_json=cluster_output,
    )
    _write_json(paths.tier_count, tier_output)

    _update_status(paths, "merge")

    # 4) Merge ClusterForge + TierCount
    merged_output = merge_cluster_outputs(cluster_output, tier_output)
    merged_output = _strip_raw_text(merged_output)
    _write_json(paths.cluster_merged, merged_output)

    _update_status(paths, "cluster_composer")

    # 5) ClusterComposer per cluster
    composer = ClusterComposer()
    cluster_outlines: dict[str, Any] = {}
    for cluster in merged_output.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "")
        if not cluster_id:
            continue
        cluster_result = composer.generate(
            merged_clusters=merged_output,
            cluster_id=cluster_id,
            description=guidance_text,
            special_notes="",
        )
        cluster_result = _strip_raw_text(cluster_result)
        _write_json(paths.cluster_composer(cluster_id), cluster_result)

        # outline per cluster
        outlined = compute_cluster_outline(cluster_result)
        outlined = _strip_raw_text(outlined)
        _write_json(paths.cluster_outline(cluster_id), outlined)
        cluster_outlines[cluster_id] = outlined

    _write_json(paths.cluster_outlines_all, cluster_outlines)

    _update_status(paths, "cluster_relation_planner")

    # 6) ClusterRelationPlanner (macro relations before placement)
    relation_planner = ClusterRelationPlanner()
    relation_plan = relation_planner.generate(
        room_model_json=room_output,
        clusters_json=cluster_outlines,
        description=guidance_text,
        special_notes="",
    )
    relation_plan = _strip_raw_text(relation_plan)
    _write_json(paths.cluster_relation_plan, relation_plan)

    _update_status(paths, "cluster_placer")

    # 7) Cluster placement (solver + phase-2 placer)
    filtered_cluster_outlines = _filter_unsat_cluster_outlines(cluster_outlines)
    solver = MacroClusterSolver()
    solver_output = solver.generate(
        room_model_json=room_output,
        clusters_outlines_json=filtered_cluster_outlines,
        relation_plan_json=relation_plan,
        cluster_constraints_json=cluster_output,
        grid_mm=GLOBAL_LAYOUT_GRID_MM,
    )
    solver_output = _strip_raw_text(solver_output)
    _write_json(paths.cluster_solver, solver_output)

    controller = Phase2Controller()
    phase2_result = controller.generate(
        room_interpreter_json=room_output,
        cluster_merged_json=merged_output,
        cluster_outlines_json=cluster_outlines,
        relation_plan_json=relation_plan,
        solver_output_json=solver_output,
        cluster_constraints_json=None,
    )
    placer_output = _strip_raw_text(phase2_result.get("proposal") or {})
    _write_json(paths.cluster_placer, placer_output)

    # if solver_status != "OK":
    #     _update_status(
    #         paths,
    #         "error",
    #         error=(
    #             "cluster_solver returned "
    #             f"{solver_status!s}; cannot build a valid absolute layout or run stylist."
    #         ),
    #     )
    #     return {
    #         "case_id": case_id,
    #         "case_dir": str(case_root),
    #         "final_output": None,
    #         "error": solver_status,
    #     }

    _update_status(paths, "absolute_layout")

    # 8) Absolute layout (from controller-selected proposal)
    absolute_layout = _strip_raw_text(phase2_result.get("absolute_layout") or {})
    absolute_layout = merge_manual_placements_into_absolute_layout(
        absolute_layout,
        manual_placements,
    )
    _write_json(paths.absolute_layout, absolute_layout)

    absolute_status = (
        absolute_layout.get("status") if isinstance(absolute_layout, dict) else None
    )
    if absolute_status != "OK":
        missing = (
            absolute_layout.get("missing", [])
            if isinstance(absolute_layout, dict)
            else []
        )
        missing_preview = missing[:8] if isinstance(missing, list) else missing
        _update_status(
            paths,
            "error",
            error=(
                "absolute_layout returned "
                f"{absolute_status!s}; missing={json.dumps(missing_preview, ensure_ascii=True)}"
            ),
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": absolute_status,
        }

    _update_status(paths, "stylist")

    # 9) Stylist
    stylist = Stylist()
    room_meta = room_output.get("meta") if isinstance(room_output, dict) else None
    stylist_tenant_id = _extract_tenant_id_from_payload(prepared_input_payload)
    stylist_user_context = {
        "user_id": user_id,
        "tenant_id": stylist_tenant_id,
        "style": room_meta.get("style") if isinstance(room_meta, dict) else "",
        "room_notes": room_output.get("notes", [])
        if isinstance(room_output, dict)
        else [],
        "guidance_text": guidance_text,
        "user_input": prepared_input_payload.get("user_input", {})
        if isinstance(prepared_input_payload, dict)
        else {},
    }
    style_plan = stylist.generate_style_plan(
        layout_json=absolute_layout,
        user_context_json=stylist_user_context,
        tenant_id=stylist_tenant_id,
    )
    stylist_output = stylist.apply_style_plan(
        layout_json=absolute_layout,
        user_context_json=stylist_user_context,
        tenant_id=stylist_tenant_id,
        style_plan=style_plan,
    )
    stylist_output = _strip_raw_text(stylist_output)
    stylist_output = merge_manual_placements_into_styled_output(
        stylist_output,
        manual_placements,
    )
    _write_json(paths.stylist, stylist_output)

    stylist_status = (
        stylist_output.get("status") if isinstance(stylist_output, dict) else None
    )
    if stylist_status != "OK":
        missing = (
            stylist_output.get("missing", [])
            if isinstance(stylist_output, dict)
            else []
        )
        missing_preview = missing[:8] if isinstance(missing, list) else missing
        _update_status(
            paths,
            "error",
            error=(
                "stylist returned "
                f"{stylist_status!s}; missing={json.dumps(missing_preview, ensure_ascii=True)}"
            ),
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": stylist_output,
            "error": stylist_status,
        }

    _update_status(paths, "layout_variants")
    layout_variants = build_layout_variants_artifact(
        room_output=room_output,
        merged_output=merged_output,
        cluster_outlines=cluster_outlines,
        relation_plan=relation_plan,
        solver_output=solver_output,
        phase2_result=phase2_result,
        absolute_layout=absolute_layout,
        stylist_output=stylist_output,
        stylist=stylist,
        stylist_user_context_json=stylist_user_context,
        stylist_tenant_id=stylist_tenant_id,
        style_plan=style_plan,
        manual_placements=manual_placements,
        max_variants=5,
    )
    layout_variants = _strip_raw_text(layout_variants)
    layout_variants = merge_manual_placements_into_layout_variants(
        layout_variants,
        manual_placements,
    )
    _write_json(paths.layout_variants, layout_variants)

    _update_status(paths, "done")

    return {
        "case_id": case_id,
        "case_dir": str(case_root),
        "final_output": stylist_output,
    }


def case_paths(case_id: str, cases_root: str | Path = "cases") -> CasePaths:
    return CasePaths(case_id=case_id, root=Path(cases_root) / case_id)


def _make_case_id(user_id: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "user")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_user}_{ts}"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_room_type(
    input_payload: dict[str, Any], room_output: dict[str, Any]
) -> str:
    user_input = (
        input_payload.get("user_input") if isinstance(input_payload, dict) else None
    )
    if isinstance(user_input, dict):
        rt = user_input.get("room_type")
        if isinstance(rt, str) and rt:
            return rt
    meta = room_output.get("meta") if isinstance(room_output, dict) else None
    if isinstance(meta, dict):
        rt = meta.get("room_type")
        if isinstance(rt, str) and rt:
            return rt
    return "unknown"


def _extract_tenant_id_from_payload(input_payload: dict[str, Any]) -> str | None:
    tenant_id = input_payload.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        tenant_id = user_input.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id.strip():
            return tenant_id.strip()
    return None


def _strip_raw_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload.pop("raw_text", None)
        for key, value in list(payload.items()):
            payload[key] = _strip_raw_text(value)
        return payload
    if isinstance(payload, list):
        return [_strip_raw_text(item) for item in payload]
    return payload


def _filter_unsat_cluster_outlines(
    clusters_outlines: dict[str, Any] | list[Any],
) -> dict[str, Any] | list[Any]:
    removed_ids: list[str] = []

    def is_unsat_status(value: Any) -> bool:
        return isinstance(value, str) and value.strip().upper() == "UNSAT"

    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            kept = [
                item
                for item in clusters
                if not (isinstance(item, dict) and is_unsat_status(item.get("status")))
            ]
            for item in clusters:
                if isinstance(item, dict) and is_unsat_status(item.get("status")):
                    cid = item.get("cluster_id")
                    if isinstance(cid, str) and cid:
                        removed_ids.append(cid)
            out = dict(clusters_outlines)
            out["clusters"] = kept
            if removed_ids:
                logger.info(
                    "Orchestrator filtered UNSAT clusters: %s",
                    sorted(set(removed_ids)),
                )
            return out

        out: dict[str, Any] = {}
        for cid, payload in clusters_outlines.items():
            if isinstance(payload, dict) and is_unsat_status(payload.get("status")):
                if isinstance(cid, str) and cid:
                    removed_ids.append(cid)
                continue
            out[cid] = payload
        if removed_ids:
            logger.info(
                "Orchestrator filtered UNSAT clusters: %s",
                sorted(set(removed_ids)),
            )
        return out

    if isinstance(clusters_outlines, list):
        kept_list = [
            item
            for item in clusters_outlines
            if not (isinstance(item, dict) and is_unsat_status(item.get("status")))
        ]
        for item in clusters_outlines:
            if isinstance(item, dict) and is_unsat_status(item.get("status")):
                cid = item.get("cluster_id")
                if isinstance(cid, str) and cid:
                    removed_ids.append(cid)
        if removed_ids:
            logger.info(
                "Orchestrator filtered UNSAT clusters: %s",
                sorted(set(removed_ids)),
            )
        return kept_list

    return clusters_outlines


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _update_status(paths: CasePaths, stage: str, *, error: str | None = None) -> None:
    payload: dict[str, Any] = {
        "case_id": paths.case_id,
        "stage": stage,
        "updated_at_utc": _now_utc_iso(),
    }
    if error is not None:
        payload["error"] = str(error)
    _write_json(paths.status, payload)
