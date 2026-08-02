from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    root: Path

    @property
    def room_interpreter(self) -> Path:
        return self.root / "01_room_interpreter.json"

    @property
    def seed_relation_planner_legacy(self) -> Path:
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

    @property
    def module_outputs_dir(self) -> Path:
        return self.root / "module_outputs"

    @property
    def module_io_manifest(self) -> Path:
        return self.root / "module_io_manifest.json"

    def module_output(self, module_name: str) -> Path:
        return self.module_outputs_dir / f"{module_name}.json"

    @property
    def cluster_relation_plan(self) -> Path:
        return self.root / "05b_cluster_relation_plan.json"

    @property
    def seed_relation_plans(self) -> Path:
        return self.root / "05c_seed_relation_plans.json"

    @property
    def cluster_solver(self) -> Path:
        return self.root / "06a_cluster_solver.json"

    @property
    def solver_dropped_inventory(self) -> Path:
        return self.root / "06b_solver_dropped_inventory.json"

    @property
    def absolute_layout(self) -> Path:
        return self.root / "07_absolute_layout.json"

    @property
    def accessory_refill(self) -> Path:
        return self.root / "07b_accessory_refill.json"

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


@dataclass(frozen=True)
class OrchestratorModuleSpec:
    name: str
    implementation: str
    inputs: tuple[str, ...]
    output: str
    output_artifact: str
    legacy_artifact: str | None = None


ORCHESTRATOR_MODULE_SPECS: tuple[OrchestratorModuleSpec, ...] = (
    OrchestratorModuleSpec(
        name="room_interpreter",
        implementation="agent.room_interpreter.RoomInterpreter.generate",
        inputs=("prepared_input_payload", "description", "special_notes"),
        output="room model with normalized geometry, openings, room metadata, notes, and conflicts",
        output_artifact="module_outputs/room_interpreter.json",
        legacy_artifact="01_room_interpreter.json",
    ),
    OrchestratorModuleSpec(
        name="stylist_style_policy",
        implementation="agent.stylist.Stylist.compile_style_policy",
        inputs=("room_type", "planning_guidance_text", "room_output"),
        output="style policy used by semantic planning and final styling",
        output_artifact="module_outputs/stylist_style_policy.json",
    ),
    OrchestratorModuleSpec(
        name="seed_relation_planner",
        implementation="agent.seed_relation_planner.SeedRelationPlanner.generate",
        inputs=(
            "room_type",
            "planning_guidance_text",
            "room_output",
            "inventory_catalog",
            "style_policy",
        ),
        output="semantic layout program and cluster definitions with anchor-first policies",
        output_artifact="module_outputs/seed_relation_planner.json",
        legacy_artifact="02_cluster_forge.json",
    ),
    OrchestratorModuleSpec(
        name="request_contract",
        implementation="agent.request_contract.request_contract_from_payload",
        inputs=("planning_guidance_text", "cluster_output"),
        output="request-aware object contract inferred before tier count",
        output_artifact="module_outputs/request_contract.json",
    ),
    OrchestratorModuleSpec(
        name="tier_count_director",
        implementation="agent.tier_count_director.TierCountDirector.generate",
        inputs=(
            "planning_guidance_text",
            "room_output",
            "prepared_input_payload",
            "cluster_output",
        ),
        output="inventory tier/count decisions per cluster",
        output_artifact="module_outputs/tier_count_director.json",
        legacy_artifact="03_tier_count.json",
    ),
    OrchestratorModuleSpec(
        name="cluster_output_merger",
        implementation="object_program.builder.build_solver_object_program",
        inputs=("cluster_output", "tier_output"),
        output="solver-ready merged object program with filtered anchor/support graph",
        output_artifact="module_outputs/cluster_output_merger.json",
        legacy_artifact="04_cluster_merged.json",
    ),
    OrchestratorModuleSpec(
        name="seed_concept_generator",
        implementation="agent.seed_concept_generator.SeedConceptGenerator.generate_bundle",
        inputs=("room_output", "merged_object_program", "target_count"),
        output="bundle of LLM-guided anchor-first macro concepts",
        output_artifact="module_outputs/seed_concept_generator.json",
        legacy_artifact="05c_seed_relation_plans.json",
    ),
    OrchestratorModuleSpec(
        name="seed_concept_relation_plan",
        implementation="agent.seed_concept_generator.solver_plan_from_concept",
        inputs=("macro_concept", "room_output", "room_type"),
        output="canonical relation plan consumed by the macro cluster solver",
        output_artifact="module_outputs/seed_concept_relation_plan.json",
        legacy_artifact="05b_cluster_relation_plan.json",
    ),
    OrchestratorModuleSpec(
        name="macro_cluster_solver",
        implementation="agent.solver.solver.MacroClusterSolver.generate_object_layout",
        inputs=(
            "room_output",
            "merged_object_program",
            "relation_plan",
            "cluster_output",
        ),
        output="absolute macro-cluster layout solved directly from anchors and support constraints",
        output_artifact="module_outputs/macro_cluster_solver.json",
        legacy_artifact="06a_cluster_solver.json",
    ),
    OrchestratorModuleSpec(
        name="controlled_accessory_refill",
        implementation="pipeline.dropped_inventory.controlled_accessory_refill",
        inputs=(
            "room_output",
            "absolute_layout",
            "cluster_output",
            "dropped_inventory_by_cluster",
            "refill_policy",
        ),
        output="optional accessory refill summary and updated absolute layout when eligible",
        output_artifact="module_outputs/controlled_accessory_refill.json",
        legacy_artifact="07b_accessory_refill.json",
    ),
    OrchestratorModuleSpec(
        name="absolute_layout",
        implementation="agent.solver.solver.MacroClusterSolver.generate_object_layout:absolute_layout",
        inputs=("macro_cluster_solver output", "manual_placements"),
        output="canonical absolute layout consumed by Stylist",
        output_artifact="module_outputs/absolute_layout.json",
        legacy_artifact="07_absolute_layout.json",
    ),
    OrchestratorModuleSpec(
        name="stylist",
        implementation="agent.stylist.Stylist.generate_style_plan/apply_style_plan",
        inputs=("absolute_layout", "stylist_user_context", "tenant_id"),
        output="styled result with selected inventory/assets/material/style metadata",
        output_artifact="module_outputs/stylist.json",
        legacy_artifact="08_stylist.json",
    ),
    OrchestratorModuleSpec(
        name="layout_variants",
        implementation="pipeline.layout_variants.select_distinct_final_gallery_candidates",
        inputs=("final styled candidates", "target_final_count"),
        output="final selectable gallery variants",
        output_artifact="module_outputs/layout_variants.json",
        legacy_artifact="09_layout_variants.json",
    ),
)


def case_paths(case_id: str, cases_root: str | Path = "cases") -> CasePaths:
    return CasePaths(case_id=case_id, root=Path(cases_root) / case_id)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp_path.replace(path)


def _mark_downstream_module_outputs_stale(
    paths: CasePaths,
    *,
    blocked_by: str,
) -> None:
    payload = {
        "status": "STALE",
        "stale": True,
        "blocked_by": blocked_by,
        "generated_at_utc": _now_utc_iso(),
        "reason": "Run failed before this module was regenerated.",
    }
    for module_name, legacy_path in (
        ("macro_cluster_solver", paths.cluster_solver),
        ("controlled_accessory_refill", paths.accessory_refill),
        ("absolute_layout", paths.absolute_layout),
        ("stylist", paths.stylist),
        ("layout_variants", paths.layout_variants),
    ):
        _write_module_json(paths, module_name, payload, legacy_path=legacy_path)


def _write_module_json(
    paths: CasePaths,
    module_name: str,
    payload: Any,
    *,
    legacy_path: Path | None = None,
) -> None:
    _write_json(paths.module_output(module_name), payload)
    if legacy_path is not None:
        _write_json(legacy_path, payload)


def _write_module_io_manifest(paths: CasePaths) -> None:
    _write_json(paths.module_io_manifest, _module_io_manifest_payload(paths))


def _module_io_manifest_payload(paths: CasePaths) -> dict[str, Any]:
    return {
        "case_id": paths.case_id,
        "scope": "pipeline.orchestrator.run_case",
        "generated_at_utc": _now_utc_iso(),
        "modules": [
            {
                "module": spec.name,
                "implementation": spec.implementation,
                "inputs": list(spec.inputs),
                "output": spec.output,
                "output_artifact": spec.output_artifact,
                "legacy_artifact": spec.legacy_artifact,
            }
            for spec in ORCHESTRATOR_MODULE_SPECS
        ],
    }


def _update_status(
    paths: CasePaths,
    stage: str,
    *,
    error: str | None = None,
    message: str | None = None,
    progress_current: int | None = None,
    progress_total: int | None = None,
) -> None:
    existing_payload: dict[str, Any] = {}
    if paths.status.exists():
        try:
            existing_payload = json.loads(paths.status.read_text())
        except (OSError, json.JSONDecodeError):
            existing_payload = {}

    payload: dict[str, Any] = {
        "case_id": paths.case_id,
        "stage": stage,
        "updated_at_utc": _now_utc_iso(),
    }
    if error is not None:
        payload["error"] = str(error)
    if isinstance(message, str) and message.strip():
        payload["message"] = message.strip()
    if progress_current is not None:
        payload["progress_current"] = int(progress_current)
    if progress_total is not None:
        payload["progress_total"] = int(progress_total)
    actions = existing_payload.get("actions")
    action_history = (
        [item for item in actions if isinstance(item, dict)]
        if isinstance(actions, list)
        else []
    )
    action_message = (
        str(error).strip() if error is not None else str(message or stage).strip()
    )
    last_action = action_history[-1] if action_history else None
    next_action = {
        "stage": stage,
        "message": action_message,
        "updated_at_utc": payload["updated_at_utc"],
        "progress_current": payload.get("progress_current"),
        "progress_total": payload.get("progress_total"),
        "error": str(error).strip() if error is not None else None,
    }
    if not isinstance(last_action, dict) or any(
        last_action.get(key) != next_action.get(key)
        for key in (
            "stage",
            "message",
            "progress_current",
            "progress_total",
            "error",
        )
    ):
        action_history.append(next_action)
    payload["actions"] = action_history
    _write_json(paths.status, payload)
