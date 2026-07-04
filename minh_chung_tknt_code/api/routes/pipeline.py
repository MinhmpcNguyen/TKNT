from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from psycopg import Error as PsycopgError
from pydantic import BaseModel, Field, ValidationError

from api.deps import get_optional_current_user
from db.models import UserAccount
from pipeline.account_layout_autosave import build_saved_layout_option_drafts
from pipeline.orchestrator import _make_case_id, run_case
from pipeline.orchestrator_artifacts import _now_utc_iso, _write_json, case_paths
from services.user_content_service import (
    LayoutOptionPersistenceRequest,
    UserContentService,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)

class PipelineRunRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    input_payload: dict[str, Any]
    case_name: str | None = None
    description: str | None = None
    special_notes: str | None = None


class PipelineRunResponse(BaseModel):
    case_id: str
    case_dir: str
    status_path: str
    status: str


class PipelineStatusResponse(BaseModel):
    case_id: str
    stage: str
    updated_at_utc: str
    error: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)


class PipelineResultResponse(BaseModel):
    case_id: str
    result: dict[str, Any]


class ArtifactResponse(BaseModel):
    case_id: str
    name: str
    payload: dict[str, Any]


def _enrich_rotation_ccw(
    *,
    stylist_payload: dict[str, Any],
    absolute_layout_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    UI needs normalized orientation metadata per object.
    Stylist output intentionally omits it; we stitch it from absolute_layout,
    which contains the final semantic rotation and front-facing direction.
    """
    if not isinstance(stylist_payload, dict):
        return stylist_payload
    if not absolute_layout_payload or not isinstance(absolute_layout_payload, dict):
        return stylist_payload

    abs_objects = absolute_layout_payload.get("objects") or []
    if not isinstance(abs_objects, list) or not abs_objects:
        return stylist_payload

    by_id: dict[str, dict[str, Any]] = {}
    by_bbox: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for ao in abs_objects:
        if not isinstance(ao, dict):
            continue
        oid = ao.get("object_id")
        rot = ao.get("rotation_ccw", ao.get("rot"))
        bbox = ao.get("bbox") or {}
        if not isinstance(oid, str) or not oid:
            continue
        try:
            rotation_ccw = int(rot) % 360
        except (TypeError, ValueError):
            continue
        orientation_payload = {
            "rotation_ccw": rotation_ccw,
            "front_world": deepcopy(ao.get("front_world"))
            if isinstance(ao.get("front_world"), dict)
            else None,
            "front_side_world": (
                str(ao.get("front_side_world")).strip().lower()
                if isinstance(ao.get("front_side_world"), str)
                and str(ao.get("front_side_world")).strip()
                else None
            ),
            "axis_world": deepcopy(ao.get("axis_world"))
            if isinstance(ao.get("axis_world"), dict)
            else None,
        }
        by_id[oid] = orientation_payload
        # Best-effort fallback matching: bbox equality.
        # This helps if stylist instance ids don't exactly match absolute layout ids.
        try:
            key = (
                int(bbox.get("min_x", 0)),
                int(bbox.get("min_y", 0)),
                int(bbox.get("max_x", 0)),
                int(bbox.get("max_y", 0)),
            )
            by_bbox[key] = orientation_payload
        except (TypeError, ValueError):
            pass

    out = dict(stylist_payload)
    objects = out.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return out

    enriched: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        inst = obj.get("instance_id") or obj.get("id")
        bbox = obj.get("bbox") or {}

        orientation_payload: dict[str, Any] | None = None
        if isinstance(inst, str) and inst in by_id:
            orientation_payload = by_id[inst]
        else:
            # bbox match (best-effort)
            try:
                key = (
                    int(bbox.get("min_x", 0)),
                    int(bbox.get("min_y", 0)),
                    int(bbox.get("max_x", 0)),
                    int(bbox.get("max_y", 0)),
                )
                if key in by_bbox:
                    orientation_payload = by_bbox[key]
            except (TypeError, ValueError):
                orientation_payload = None

        next_obj = dict(obj)
        next_obj["rotation_ccw"] = (
            int(
                (orientation_payload or {}).get(
                    "rotation_ccw",
                    next_obj.get("rotation_ccw", next_obj.get("rot", 0)),
                )
                or 0
            )
            % 360
        )
        next_obj["front_world"] = deepcopy(
            (orientation_payload or {}).get("front_world")
        )
        next_obj["front_side_world"] = (orientation_payload or {}).get(
            "front_side_world"
        )
        next_obj["axis_world"] = deepcopy((orientation_payload or {}).get("axis_world"))
        enriched.append(next_obj)

    out["objects"] = enriched
    return out


def _run_in_thread(**kwargs: Any) -> None:
    account_user_id = kwargs.pop("account_user_id", None)
    account_case_name = kwargs.pop("account_case_name", None)
    try:
        run_case(**kwargs)
        if isinstance(account_user_id, str) and account_user_id.strip():
            try:
                _autosave_pipeline_case(
                    account_user_id=account_user_id,
                    case_id=str(kwargs["case_id"]),
                    cases_root=str(kwargs["cases_root"]),
                    case_name=account_case_name
                    if isinstance(account_case_name, str)
                    else None,
                )
            except (
                ValueError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeError,
                PsycopgError,
                ValidationError,
            ) as exc:
                logger.exception(
                    "Pipeline account autosave failed for case %s: %s",
                    kwargs.get("case_id"),
                    exc,
                )
                _append_status_action(
                    case_id=str(kwargs["case_id"]),
                    cases_root=str(kwargs["cases_root"]),
                    stage="account_autosave_warning",
                    message=f"Generated layouts were not saved to account: {exc}",
                    error=str(exc),
                )
    except Exception as exc:  # noqa: BLE001
        paths = case_paths(kwargs["case_id"], kwargs["cases_root"])
        existing_payload: dict[str, Any] = {}
        if paths.status.exists():
            try:
                existing_payload = json.loads(paths.status.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                existing_payload = {}
        actions = existing_payload.get("actions")
        action_history = (
            [item for item in actions if isinstance(item, dict)]
            if isinstance(actions, list)
            else []
        )
        updated_at_utc = _now_utc_iso()
        action_history.append(
            {
                "stage": "error",
                "message": str(exc),
                "updated_at_utc": updated_at_utc,
                "progress_current": existing_payload.get("progress_current"),
                "progress_total": existing_payload.get("progress_total"),
                "error": str(exc),
            }
        )
        _write_json(
            paths.status,
            {
                "case_id": paths.case_id,
                "stage": "error",
                "updated_at_utc": updated_at_utc,
                "error": str(exc),
                "actions": action_history,
            },
        )


def _autosave_pipeline_case(
    *,
    account_user_id: str,
    case_id: str,
    cases_root: str,
    case_name: str | None,
) -> None:
    service = UserContentService()
    user = service.get_user_by_id(account_user_id)
    if user is None:
        raise ValueError(f"Cannot autosave layouts for unknown user {account_user_id}.")

    paths = case_paths(case_id, cases_root)
    resolved_case_name = _resolve_account_case_name(case_name, case_id)
    drafts = build_saved_layout_option_drafts(
        case_id=case_id,
        case_name=resolved_case_name,
        room_payload=_load_json_payload(paths.room_interpreter),
        layout_variants_payload=_load_json_payload(paths.layout_variants),
    )
    if not drafts:
        _append_status_action(
            case_id=case_id,
            cases_root=cases_root,
            stage="account_autosave_warning",
            message="No generated layout options were available for account autosave.",
            error=None,
        )
        return

    saved = service.upsert_layout_case_options(
        user=user,
        case_id=case_id,
        case_name=resolved_case_name,
        options=[
            LayoutOptionPersistenceRequest(
                option_id=draft.option_id,
                option_index=draft.option_index,
                name=draft.name,
                floorplan_json=draft.floorplan_json,
                design_json=draft.design_json,
                styled_result_json=draft.styled_result_json,
                meta=draft.meta,
            )
            for draft in drafts
        ],
        meta={
            "source": "pipeline_autosave",
            "source_pipeline_case_id": case_id,
            "option_count": len(drafts),
        },
    )
    _append_status_action(
        case_id=case_id,
        cases_root=cases_root,
        stage="account_autosave",
        message=(
            f"Saved {len(saved)} generated layout option"
            f"{'' if len(saved) == 1 else 's'} to account case."
        ),
        error=None,
    )


def _resolve_account_case_name(case_name: str | None, case_id: str) -> str:
    if case_name is not None and case_name.strip():
        return case_name.strip()
    return f"Layout case {case_id}"


def _load_json_payload(path: Any) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _append_status_action(
    *,
    case_id: str,
    cases_root: str,
    stage: str,
    message: str,
    error: str | None,
) -> None:
    paths = case_paths(case_id, cases_root)
    payload: dict[str, Any] = {}
    if paths.status.exists():
        try:
            payload = json.loads(paths.status.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
    actions = payload.get("actions")
    action_history = (
        [item for item in actions if isinstance(item, dict)]
        if isinstance(actions, list)
        else []
    )
    updated_at_utc = _now_utc_iso()
    action_history.append(
        {
            "stage": stage,
            "message": message,
            "updated_at_utc": updated_at_utc,
            "progress_current": payload.get("progress_current"),
            "progress_total": payload.get("progress_total"),
            "error": error,
        }
    )
    payload.update(
        {
            "case_id": case_id,
            "stage": payload.get("stage", "done"),
            "updated_at_utc": updated_at_utc,
            "actions": action_history,
        }
    )
    _write_json(paths.status, payload)


@router.post("/run", response_model=PipelineRunResponse)
def run_pipeline(
    req: PipelineRunRequest,
    background: BackgroundTasks,
    current_user: UserAccount | None = Depends(get_optional_current_user),
) -> PipelineRunResponse:
    pipeline_user_id = str(current_user.id) if current_user is not None else req.user_id
    case_id = _make_case_id(pipeline_user_id)
    paths = case_paths(case_id)
    paths.root.mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.status,
        {
            "case_id": case_id,
            "stage": "queued",
            "updated_at_utc": _now_utc_iso(),
        },
    )

    background.add_task(
        _run_in_thread,
        input_payload=req.input_payload,
        user_id=pipeline_user_id,
        description=req.description,
        special_notes=req.special_notes,
        cases_root=str(paths.root.parent),
        case_id=case_id,
        account_user_id=str(current_user.id) if current_user is not None else None,
        account_case_name=req.case_name or req.description,
    )

    return PipelineRunResponse(
        case_id=case_id,
        case_dir=str(paths.root),
        status_path=str(paths.status),
        status="queued",
    )


@router.get("/{case_id}/status", response_model=PipelineStatusResponse)
def get_status(case_id: str) -> PipelineStatusResponse:
    paths = case_paths(case_id)
    if not paths.status.exists():
        raise HTTPException(status_code=404, detail="case not found")
    payload = json.loads(paths.status.read_text())
    return PipelineStatusResponse(
        case_id=payload.get("case_id", case_id),
        stage=payload.get("stage", "unknown"),
        updated_at_utc=payload.get("updated_at_utc", ""),
        error=payload.get("error"),
        message=payload.get("message"),
        progress_current=payload.get("progress_current"),
        progress_total=payload.get("progress_total"),
        actions=payload.get("actions")
        if isinstance(payload.get("actions"), list)
        else [],
    )


@router.get("/{case_id}/result", response_model=PipelineResultResponse)
def get_result(case_id: str) -> PipelineResultResponse:
    paths = case_paths(case_id)
    if not paths.stylist.exists() and not paths.layout_variants.exists():
        raise HTTPException(status_code=404, detail="result not found")
    enriched: dict[str, Any] = {}
    if paths.stylist.exists():
        stylist_payload = json.loads(paths.stylist.read_text())
        absolute_payload: dict[str, Any] | None = None
        try:
            if paths.absolute_layout.exists():
                absolute_payload = json.loads(paths.absolute_layout.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            absolute_payload = None

        enriched = _enrich_rotation_ccw(
            stylist_payload=stylist_payload,
            absolute_layout_payload=absolute_payload,
        )
    if paths.layout_variants.exists():
        variants_payload = json.loads(paths.layout_variants.read_text())
        if isinstance(variants_payload, dict):
            variants = variants_payload.get("variants")
            selected_variant_id = variants_payload.get("selected_variant_id")
            if isinstance(variants, list):
                enriched_variants: list[dict[str, Any]] = []
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    styled_variant = variant.get("styled_result")
                    absolute_variant = variant.get("absolute_layout")
                    next_variant = dict(variant)
                    if isinstance(styled_variant, dict):
                        next_variant["styled_result"] = _enrich_rotation_ccw(
                            stylist_payload=styled_variant,
                            absolute_layout_payload=absolute_variant
                            if isinstance(absolute_variant, dict)
                            else None,
                        )
                    enriched_variants.append(next_variant)
                if enriched_variants and not enriched:
                    first_variant = enriched_variants[0]
                    styled_result = first_variant.get("styled_result")
                    if isinstance(styled_result, dict):
                        enriched = dict(styled_result)
                enriched["variants"] = enriched_variants
            if isinstance(selected_variant_id, str) and selected_variant_id:
                enriched["selected_variant_id"] = selected_variant_id
    return PipelineResultResponse(case_id=case_id, result=enriched)


@router.get("/{case_id}/artifact/{name}", response_model=ArtifactResponse)
def get_artifact(case_id: str, name: str) -> ArtifactResponse:
    paths = case_paths(case_id)
    mapping = {
        "module_io_manifest": paths.module_io_manifest,
        "room_interpreter": paths.room_interpreter,
        "stylist_style_policy": paths.module_output("stylist_style_policy"),
        "seed_relation_planner": paths.module_output("seed_relation_planner"),
        "tier_count": paths.tier_count,
        "tier_count_director": paths.module_output("tier_count_director"),
        "cluster_merged": paths.cluster_merged,
        "cluster_output_merger": paths.module_output("cluster_output_merger"),
        "cluster_relation_plan": paths.cluster_relation_plan,
        "seed_concept_relation_plan": paths.module_output("seed_concept_relation_plan"),
        "seed_concept_generator": paths.module_output("seed_concept_generator"),
        "macro_cluster_solver": paths.module_output("macro_cluster_solver"),
        "macro_cluster_solver_dropped_inventory": paths.module_output(
            "macro_cluster_solver_dropped_inventory"
        ),
        "absolute_layout": paths.absolute_layout,
        "controlled_accessory_refill": paths.module_output(
            "controlled_accessory_refill"
        ),
        "stylist": paths.stylist,
        "layout_variants": paths.layout_variants,
    }
    path = mapping.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(case_id=case_id, name=name, payload=payload)
