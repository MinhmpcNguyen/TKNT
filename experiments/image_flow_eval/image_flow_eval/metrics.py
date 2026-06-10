from __future__ import annotations

import base64
import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import httpx
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from .types import (
    AlignMode,
    CaseConfig,
    EditOperationConfig,
    JSONValue,
    ManifestConfig,
    MetricDirection,
    MetricResult,
    MetricStatus,
    VariantConfig,
    ProviderResult,
)

_LAYOUT_FIDELITY_MODEL_ENV = "TKNT_IMAGE_FLOW_EVAL_MODEL"
_LAYOUT_FIDELITY_BASE_URL_ENV = "TKNT_IMAGE_FLOW_EVAL_OPENAI_BASE_URL"
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_IMAGE_API_KEY_ENV = "OPENAI_IMAGE_API_KEY"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_METRIC_DEVICE_ENV = "TKNT_IMAGE_FLOW_METRIC_DEVICE"
_DINO_MODEL_ENV = "TKNT_IMAGE_FLOW_DINO_MODEL"
_DEFAULT_DINO_MODEL = "facebook/dinov2-small"
_DEPTH_MODEL_ENV = "TKNT_IMAGE_FLOW_DEPTH_MODEL"
_DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_LPIPS_NET_ENV = "TKNT_IMAGE_FLOW_LPIPS_NET"
_DEFAULT_LPIPS_NET = "alex"
_YOLO_MODEL_ENV = "TKNT_IMAGE_FLOW_YOLO_MODEL"
_DEFAULT_YOLO_MODEL = "yolo11n.pt"
_OBJECT_METRIC_MAX_OBJECTS = 16
_OBJECT_METRIC_MIN_BBOX_PX = 4
_OBJECT_METRIC_CROP_PADDING_RATIO = 0.08
_DETECTOR_MIN_CONFIDENCE = 0.15
_SPATIAL_RELATION_MIN_DELTA = 0.08
_LAYOUT_GATE_WEIGHTS: Mapping[str, float] = {
    "layout_fidelity": 0.58,
    "layout_edge_structure": 0.12,
    "layout_depth_structure": 0.10,
    "layout_dino_patch_similarity": 0.08,
    "layout_dino_object_similarity": 0.04,
    "detector_layout": 0.03,
    "detector_spatial_relations": 0.02,
    "preset_fidelity": 0.03,
}
_STRUCTURAL_LAYOUT_WEIGHTS: Mapping[str, float] = {
    "layout_edge_structure": 0.35,
    "layout_depth_structure": 0.30,
    "layout_dino_patch_similarity": 0.25,
    "layout_dino_object_similarity": 0.10,
}
_HARD_FAILURE_STRICT_CAPS: Mapping[str, float] = {
    "camera_framing": 0.20,
    "object_inventory": 0.20,
    "room_shell": 0.30,
    "openings": 0.30,
    "object_position": 0.30,
    "object_scale": 0.30,
}
_HARD_FAILURE_DEFAULT_CAP = 0.35
_EDGE_DILATION_PIXELS = 3
_EDGE_PERCENTILE = 82.0

FloatArray: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]
BoolMask: TypeAlias = NDArray[np.bool_]
FidelityEvaluationKind = Literal["create", "edit"]

_CREATE_HARD_CHECKS = (
    "camera_framing",
    "room_shell",
    "openings",
    "object_inventory",
    "object_position",
    "object_scale",
    "object_silhouette",
    "unauthorized_additions",
)
_EDIT_HARD_CHECKS = (
    "camera_framing",
    "room_shell",
    "openings",
    "edit_instruction",
    "target_locality",
    "outside_target_preservation",
)


@dataclass(frozen=True)
class PairImages:
    reference: Image.Image
    candidate: Image.Image
    details: Mapping[str, JSONValue]


@dataclass(frozen=True)
class VisibleObjectSpec:
    object_id: str
    object_type: str | None
    detector_label: str | None
    bbox_norm: Mapping[str, float]
    bbox_px: Mapping[str, float]


@dataclass(frozen=True)
class DetectionBox:
    label: str
    confidence: float
    bbox_norm: Mapping[str, float]


def evaluate_variant_images(
    *,
    manifest: ManifestConfig,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    output_dir: Path,
    provider_result: ProviderResult | None = None,
) -> list[MetricResult]:
    results: list[MetricResult] = []
    metrics = set(manifest.metrics)
    if provider_result is not None:
        results.extend(compute_provider_observability_metrics(provider_result))

    if "blank" in metrics:
        results.append(
            compute_blank_check(
                output_image,
                thresholds=manifest.thresholds,
            )
        )

    if case.reference_image is not None:
        results.extend(
            compute_pair_metrics(
                prefix="reference",
                reference_path=case.reference_image,
                candidate_path=output_image,
                mask_path=None,
                selected_region="all",
                output_dir=output_dir,
                metrics=metrics,
                thresholds=manifest.thresholds,
                visual_delta_threshold=manifest.visual_delta_threshold,
                align_mode=manifest.align_mode,
                case_id=case.id,
                variant_id=variant.id,
            )
        )

    if case.source_image is not None and case.mask_image is not None:
        results.extend(
            compute_pair_metrics(
                prefix="outside_mask_preservation",
                reference_path=case.source_image,
                candidate_path=output_image,
                mask_path=case.mask_image,
                selected_region="outside_mask",
                output_dir=output_dir,
                metrics=metrics,
                thresholds=manifest.thresholds,
                visual_delta_threshold=manifest.visual_delta_threshold,
                align_mode=manifest.align_mode,
                case_id=case.id,
                variant_id=variant.id,
            )
        )

    if case.target_image is not None:
        target_region = "inside_mask" if case.mask_image is not None else "all"
        results.extend(
            compute_pair_metrics(
                prefix="target",
                reference_path=case.target_image,
                candidate_path=output_image,
                mask_path=case.mask_image,
                selected_region=target_region,
                output_dir=output_dir,
                metrics=metrics,
                thresholds=manifest.thresholds,
                visual_delta_threshold=manifest.visual_delta_threshold,
                align_mode=manifest.align_mode,
                case_id=case.id,
                variant_id=variant.id,
            )
        )

    if "layout_fidelity" in metrics:
        if case.edit_operations:
            results.extend(
                (
                    compute_render_layout_fidelity(
                        case=case,
                        variant=variant,
                        output_dir=output_dir,
                        thresholds=manifest.thresholds,
                        provider_result=provider_result,
                    ),
                    compute_edit_instruction_fidelity(
                        case=case,
                        variant=variant,
                        output_image=output_image,
                        output_dir=output_dir,
                        thresholds=manifest.thresholds,
                        provider_result=provider_result,
                    ),
                )
            )
        else:
            results.append(
                compute_layout_fidelity(
                    case=case,
                    variant=variant,
                    output_image=output_image,
                    output_dir=output_dir,
                    thresholds=manifest.thresholds,
                    provider_result=provider_result,
                )
            )
        edit_scope_result = compute_edit_scope_fidelity(
            case=case,
            variant=variant,
            output_image=output_image,
            thresholds=manifest.thresholds,
            provider_result=provider_result,
        )
        if edit_scope_result is not None:
            results.append(edit_scope_result)

    if "preset_fidelity" in metrics:
        results.append(
            compute_preset_fidelity(
                case=case,
                variant=variant,
                output_image=output_image,
                output_dir=output_dir,
                thresholds=manifest.thresholds,
                provider_result=provider_result,
            )
        )

    local_layout_metrics = {
        "layout_lpips",
        "layout_dino_similarity",
        "layout_dino_object_similarity",
        "layout_dino_patch_similarity",
        "layout_depth_structure",
        "layout_edge_structure",
        "detector_layout",
        "detector_spatial_relations",
    }
    requested_local_metrics = metrics.intersection(local_layout_metrics)
    if requested_local_metrics:
        results.extend(
            compute_local_layout_metrics(
                requested_metrics=requested_local_metrics,
                case=case,
                variant=variant,
                output_image=output_image,
                thresholds=manifest.thresholds,
                provider_result=provider_result,
            )
        )

    if "structural_layout_score" in metrics:
        results.append(
            compute_structural_layout_score(
                metrics=results,
                thresholds=manifest.thresholds,
            )
        )

    if "layout_gate_score" in metrics:
        results.append(
            compute_layout_gate_score(
                metrics=results,
                thresholds=manifest.thresholds,
            )
        )

    return results


def compute_provider_observability_metrics(
    provider_result: ProviderResult,
) -> list[MetricResult]:
    return [
        compute_provider_runtime_metric(provider_result),
        compute_provider_token_usage_metric(provider_result),
    ]


def compute_provider_runtime_metric(provider_result: ProviderResult) -> MetricResult:
    runtime_s = provider_result.runtime_s
    if runtime_s is None:
        metadata_runtime = _provider_metadata_runtime(provider_result.metadata_path)
        runtime_s = metadata_runtime
    return MetricResult(
        name="provider_runtime",
        status="passed",
        value=runtime_s,
        direction="none",
        details={
            "runtime_s": round(runtime_s, 6) if runtime_s is not None else None,
            "provider_status": provider_result.provider_status,
            "metadata_path": str(provider_result.metadata_path)
            if provider_result.metadata_path is not None
            else None,
            "measurement": "wall_clock_provider_call",
        },
    )


def compute_provider_token_usage_metric(
    provider_result: ProviderResult,
) -> MetricResult:
    usage = _provider_token_usage(provider_result.metadata_path)
    total_tokens = usage["total_tokens"]
    return MetricResult(
        name="provider_token_usage",
        status="passed",
        value=float(total_tokens) if total_tokens > 0 else None,
        direction="none",
        details=usage,
    )


def compute_preset_fidelity(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    output_dir: Path,
    thresholds: Mapping[str, float],
    provider_result: ProviderResult | None = None,
) -> MetricResult:
    if case.preset_selection is None:
        return _skipped("preset_fidelity", "preset_selection_missing")
    preset_selection = case.preset_selection.to_mapping()
    if not preset_selection:
        return _skipped("preset_fidelity", "preset_selection_empty")

    source_payload_path = (
        provider_result.source_snapshot_payload
        if provider_result and provider_result.source_snapshot_payload is not None
        else case.snapshot_payload
    )
    source_payload = (
        _read_json_mapping(source_payload_path)
        if source_payload_path is not None and source_payload_path.exists()
        else {}
    )
    source_image = (
        provider_result.source_image
        if provider_result and provider_result.source_image is not None
        else case.snapshot_image
    )
    preset_details = _preset_selection_details(preset_selection)
    source_has_window = _source_has_window(source_payload)
    deterministic_details: dict[str, JSONValue] = {
        "candidate_image": str(output_image),
        "source_image": str(source_image) if source_image is not None else None,
        "preset_selection": dict(preset_selection),
        "preset_details": preset_details,
        "source_has_window": source_has_window,
        "scenery_applicable": _scenery_applicable(preset_selection, source_has_window),
    }

    api_key = _layout_eval_api_key()
    if not api_key:
        report = {
            **deterministic_details,
            "status": "skipped",
            "reason": "preset_fidelity_vision_api_key_missing",
        }
        _write_fidelity_report(
            output_dir,
            case.id,
            variant.id,
            "preset_fidelity_report.json",
            report,
        )
        return _skipped("preset_fidelity", "vision_api_key_missing")

    report = _evaluate_preset_fidelity_with_vision(
        source_image=source_image if source_image and source_image.exists() else None,
        candidate_image=output_image,
        preset_details=preset_details,
        deterministic_details=deterministic_details,
    )
    _write_fidelity_report(
        output_dir,
        case.id,
        variant.id,
        "preset_fidelity_report.json",
        report,
    )
    score = _optional_float(report.get("score"))
    score_value = None if math.isnan(score) else score
    if score_value is None:
        return MetricResult(
            name="preset_fidelity",
            status="failed",
            details=report,
        )
    threshold = thresholds.get("min_preset_fidelity_score")
    raw_status = str(report.get("status") or "").strip().lower()
    threshold_status = _status_for_threshold(
        value=score_value,
        threshold=threshold,
        direction="min",
    )
    status: MetricStatus
    if threshold_status == "warning":
        status = "warning"
    else:
        status = (
            "passed"
            if raw_status == "passed" and threshold_status == "passed"
            else "failed"
        )
    return MetricResult(
        name="preset_fidelity",
        status=status,
        value=score_value,
        threshold=threshold,
        direction="min",
        details=report,
    )


def compute_blank_check(
    image_path: Path,
    *,
    thresholds: Mapping[str, float],
) -> MetricResult:
    try:
        image = _open_rgba(image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="blank",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )

    array = np.asarray(image, dtype=np.uint8)
    rgb = array[:, :, :3].astype(np.float32)
    alpha = array[:, :, 3]
    width, height = image.size
    file_bytes = image_path.stat().st_size
    pixel_stddev = float(np.std(rgb))
    background = _corner_background_rgb(rgb)
    background_delta = np.max(np.abs(rgb - background), axis=2)
    non_background_ratio = float(np.mean(background_delta > 3.0))
    alpha_coverage = float(np.mean(alpha > 0))
    min_stddev = thresholds.get("min_pixel_stddev", 3.0)
    min_non_background = thresholds.get("min_non_background_ratio", 0.001)
    min_file_bytes = thresholds.get("min_file_bytes", 128.0)
    min_alpha_coverage = thresholds.get("min_alpha_coverage", 0.001)
    passed = (
        pixel_stddev >= min_stddev
        and non_background_ratio >= min_non_background
        and file_bytes >= min_file_bytes
        and alpha_coverage >= min_alpha_coverage
    )
    return MetricResult(
        name="blank",
        status="passed" if passed else "failed",
        value=pixel_stddev,
        threshold=min_stddev,
        direction="min",
        details={
            "width": width,
            "height": height,
            "file_bytes": file_bytes,
            "pixel_stddev": pixel_stddev,
            "non_background_ratio": non_background_ratio,
            "alpha_coverage": alpha_coverage,
            "min_non_background_ratio": min_non_background,
            "min_file_bytes": min_file_bytes,
            "min_alpha_coverage": min_alpha_coverage,
        },
    )


def compute_layout_fidelity(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    output_dir: Path,
    thresholds: Mapping[str, float],
    provider_result: ProviderResult | None,
) -> MetricResult:
    source_image = _layout_source_image(case, provider_result)
    source_payload = _layout_source_payload(case, provider_result)
    semantic_layout_locks = _provider_semantic_layout_locks(
        provider_result.metadata_path if provider_result else None
    )
    return _compute_fidelity_metric(
        metric_name="layout_fidelity",
        report_filename="layout_fidelity_report.json",
        evaluation_kind="create",
        reference_image=source_image,
        candidate_image=output_image,
        source_payload=source_payload,
        case=case,
        variant=variant,
        output_dir=output_dir,
        thresholds=thresholds,
        semantic_layout_locks=semantic_layout_locks,
    )


def compute_render_layout_fidelity(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_dir: Path,
    thresholds: Mapping[str, float],
    provider_result: ProviderResult | None,
) -> MetricResult:
    source_image = _layout_source_image(case, provider_result)
    source_payload = _layout_source_payload(case, provider_result)
    semantic_layout_locks = _provider_semantic_layout_locks(
        provider_result.metadata_path if provider_result else None
    )
    render_image = provider_result.render_image if provider_result else None
    approved_render_source = (
        provider_result.approved_render_source if provider_result else None
    )
    if (
        render_image is not None
        and approved_render_source is not None
        and render_image == approved_render_source
    ):
        reused_metric = _load_passed_layout_metric(
            approved_render_source.parent / "metrics.json"
        )
        if reused_metric is not None:
            details: dict[str, JSONValue] = {
                "status": "passed",
                "reason": "reused_approved_render_layout_gate",
                "approved_render_source": str(approved_render_source),
                "source_metric": reused_metric.to_json(),
            }
            _write_fidelity_report(
                output_dir,
                case.id,
                variant.id,
                "render_layout_fidelity_report.json",
                details,
            )
            return MetricResult(
                name="render_layout_fidelity",
                status="passed",
                value=reused_metric.value,
                threshold=reused_metric.threshold,
                direction=reused_metric.direction,
                details=details,
            )
    return _compute_fidelity_metric(
        metric_name="render_layout_fidelity",
        report_filename="render_layout_fidelity_report.json",
        evaluation_kind="create",
        reference_image=source_image,
        candidate_image=render_image,
        source_payload=source_payload,
        case=case,
        variant=variant,
        output_dir=output_dir,
        thresholds=thresholds,
        semantic_layout_locks=semantic_layout_locks,
    )


def _load_passed_layout_metric(path: Path) -> MetricResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        if item.get("name") != "layout_fidelity" or item.get("status") != "passed":
            continue
        direction_raw = str(item.get("direction") or "none")
        direction = (
            cast(MetricDirection, direction_raw)
            if direction_raw in {"min", "max", "none"}
            else "none"
        )
        return MetricResult(
            name="layout_fidelity",
            status="passed",
            value=_optional_number(item.get("value")),
            threshold=_optional_number(item.get("threshold")),
            direction=direction,
            details=_json_value_mapping(item.get("details")),
        )
    return None


def compute_edit_instruction_fidelity(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    output_dir: Path,
    thresholds: Mapping[str, float],
    provider_result: ProviderResult | None,
) -> MetricResult:
    render_image = provider_result.render_image if provider_result else None
    source_payload = _layout_source_payload(case, provider_result)
    semantic_layout_locks = _provider_semantic_layout_locks(
        provider_result.metadata_path if provider_result else None
    )
    return _compute_fidelity_metric(
        metric_name="edit_instruction_fidelity",
        report_filename="edit_instruction_fidelity_report.json",
        evaluation_kind="edit",
        reference_image=render_image,
        candidate_image=output_image,
        source_payload=source_payload,
        case=case,
        variant=variant,
        output_dir=output_dir,
        thresholds=thresholds,
        semantic_layout_locks=semantic_layout_locks,
    )


def _compute_fidelity_metric(
    *,
    metric_name: str,
    report_filename: str,
    evaluation_kind: FidelityEvaluationKind,
    reference_image: Path | None,
    candidate_image: Path | None,
    source_payload: Mapping[str, object],
    case: CaseConfig,
    variant: VariantConfig,
    output_dir: Path,
    thresholds: Mapping[str, float],
    semantic_layout_locks: Sequence[str] = (),
) -> MetricResult:
    if reference_image is None:
        return MetricResult(
            name=metric_name,
            status="failed",
            details={"reason": "reference_image_missing"},
        )
    if candidate_image is None:
        return MetricResult(
            name=metric_name,
            status="failed",
            details={"reason": "candidate_image_missing"},
        )
    try:
        source = _open_rgba(reference_image).convert("RGB")
        candidate = _open_rgba(candidate_image).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name=metric_name,
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )

    source_aspect = source.width / max(1, source.height)
    candidate_aspect = candidate.width / max(1, candidate.height)
    aspect_delta = abs(source_aspect - candidate_aspect) / max(source_aspect, 1e-6)
    max_aspect_delta = thresholds.get("max_layout_aspect_delta", 0.015)
    deterministic_details: dict[str, JSONValue] = {
        "evaluation_kind": evaluation_kind,
        "source_image": str(reference_image),
        "candidate_image": str(candidate_image),
        "source_size": [source.width, source.height],
        "candidate_size": [candidate.width, candidate.height],
        "source_aspect": round(source_aspect, 6),
        "candidate_aspect": round(candidate_aspect, 6),
        "aspect_delta": round(aspect_delta, 6),
        "max_layout_aspect_delta": max_aspect_delta,
        "visible_objects": _visible_object_details(source_payload),
        "openings": _opening_details(source_payload),
        "edit_operations": _edit_operation_details(case),
        "semantic_layout_locks": list(semantic_layout_locks),
    }
    if aspect_delta > max_aspect_delta:
        report = {
            **deterministic_details,
            "status": "failed",
            "reason": "source_candidate_aspect_mismatch",
        }
        _write_fidelity_report(output_dir, case.id, variant.id, report_filename, report)
        return MetricResult(
            name=metric_name,
            status="failed",
            value=aspect_delta,
            threshold=max_aspect_delta,
            direction="max",
            details=report,
        )

    required_hard_checks = (
        _CREATE_HARD_CHECKS if evaluation_kind == "create" else _EDIT_HARD_CHECKS
    )
    vision_result = _evaluate_fidelity_with_vision(
        source_image=reference_image,
        candidate_image=candidate_image,
        source_payload=source_payload,
        case=case,
        variant=variant,
        deterministic_details=deterministic_details,
        evaluation_kind=evaluation_kind,
        required_hard_checks=required_hard_checks,
        semantic_layout_locks=semantic_layout_locks,
    )
    hard_check_failures = _hard_check_failures(vision_result, required_hard_checks)
    raw_score = _optional_float(vision_result.get("score"))
    raw_score_value = None if math.isnan(raw_score) else raw_score
    score_cap = _layout_failure_cap(hard_check_failures)
    effective_score = _effective_layout_score(raw_score_value, score_cap)
    normalized_result = {
        **vision_result,
        "raw_status": vision_result.get("status"),
        "raw_score": raw_score_value,
        "effective_score": effective_score,
        "score_cap": score_cap,
        "required_hard_checks": list(required_hard_checks),
        "hard_check_failures": hard_check_failures,
        "status": "passed" if not hard_check_failures else "failed",
        "score": effective_score,
    }
    _write_fidelity_report(
        output_dir, case.id, variant.id, report_filename, normalized_result
    )
    status = str(normalized_result.get("status") or "failed")
    min_score = thresholds.get("min_layout_fidelity_score", 0.82)
    passed = (
        status == "passed"
        and effective_score is not None
        and effective_score >= min_score
    )
    return MetricResult(
        name=metric_name,
        status="passed" if passed else "failed",
        value=effective_score,
        threshold=min_score,
        direction="min",
        details=normalized_result,
    )


def compute_edit_scope_fidelity(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    thresholds: Mapping[str, float],
    provider_result: ProviderResult | None,
) -> MetricResult | None:
    if not case.edit_operations:
        return None
    render_image = provider_result.render_image if provider_result else None
    if render_image is None or not render_image.is_file():
        return MetricResult(
            name="edit_scope_fidelity",
            status="failed",
            details={"reason": "approved_render_source_missing"},
        )
    region_operations = [
        operation
        for operation in case.edit_operations
        if operation.bbox_norm is not None
    ]
    if not region_operations:
        return MetricResult(
            name="edit_scope_fidelity",
            status="skipped",
            details={"reason": "edit_operation_bbox_missing"},
        )
    try:
        source = _open_rgba(render_image).convert("RGB")
        candidate = _open_rgba(output_image).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="edit_scope_fidelity",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    if candidate.size != source.size:
        candidate = candidate.resize(source.size, Image.Resampling.BILINEAR)
    bbox_margin_ratio = thresholds.get("edit_bbox_margin_ratio", 0.015)
    editable_mask = _edit_bbox_mask(
        source.size,
        region_operations,
        margin_ratio=bbox_margin_ratio,
    )
    source_rgb = _rgb_array(source)
    candidate_rgb = _rgb_array(candidate)
    diff = np.max(
        np.abs(source_rgb.astype(np.int16) - candidate_rgb.astype(np.int16)), axis=2
    )
    outside_mask = np.asarray(editable_mask, dtype=np.uint8) == 0
    outside_change_ratio = (
        float(np.mean(diff[outside_mask] >= 22)) if outside_mask.any() else 0.0
    )
    max_outside_change_ratio = thresholds.get("max_edit_outside_change_ratio", 0.08)
    details: dict[str, JSONValue] = {
        "render_source": str(render_image),
        "candidate_image": str(output_image),
        "operation_types": [
            operation.operation_type or "prompt" for operation in case.edit_operations
        ],
        "region_count": len(region_operations),
        "outside_change_ratio": round(outside_change_ratio, 6),
        "max_edit_outside_change_ratio": max_outside_change_ratio,
        "edit_bbox_margin_ratio": bbox_margin_ratio,
    }
    return MetricResult(
        name="edit_scope_fidelity",
        status="passed"
        if outside_change_ratio <= max_outside_change_ratio
        else "failed",
        value=outside_change_ratio,
        threshold=max_outside_change_ratio,
        direction="max",
        details=details,
    )


def _layout_failure_cap(failures: Sequence[JSONValue]) -> float | None:
    failure_names = {
        str(item.get("check") or "") for item in failures if isinstance(item, Mapping)
    }
    if not failure_names:
        return None
    caps = [
        _HARD_FAILURE_STRICT_CAPS.get(name, _HARD_FAILURE_DEFAULT_CAP)
        for name in failure_names
    ]
    return min(caps) if caps else _HARD_FAILURE_DEFAULT_CAP


def _effective_layout_score(
    raw_score: float | None,
    score_cap: float | None,
) -> float | None:
    if raw_score is None:
        return None
    bounded_score = _clamp01(raw_score)
    if score_cap is None:
        return bounded_score
    return min(bounded_score, score_cap)


def _metric_by_name(
    metrics: Sequence[MetricResult],
    name: str,
) -> MetricResult | None:
    for metric in metrics:
        if metric.name == name:
            return metric
    return None


def _metric_score_value(metric: MetricResult) -> float | None:
    if metric.value is None:
        return None
    if metric.direction == "max":
        return _clamp01(1.0 - metric.value)
    return _clamp01(metric.value)


def _layout_metric_score_cap(metric: MetricResult) -> float | None:
    details = metric.details or {}
    score_cap = _optional_number(details.get("score_cap"))
    if score_cap is not None:
        return score_cap
    failures = details.get("hard_check_failures")
    if isinstance(failures, list):
        failure_cap = _layout_failure_cap(failures)
        if failure_cap is not None:
            return failure_cap
    if metric.status == "failed":
        return _HARD_FAILURE_DEFAULT_CAP
    return None


def compute_local_layout_metrics(
    *,
    requested_metrics: set[str],
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    thresholds: Mapping[str, float],
    provider_result: ProviderResult | None,
) -> list[MetricResult]:
    reference_image = _local_layout_reference_image(case, provider_result)
    source_payload = _layout_source_payload(case, provider_result)
    results: list[MetricResult] = []
    if "layout_lpips" in requested_metrics:
        results.append(
            compute_layout_lpips(
                reference_image=reference_image,
                candidate_image=output_image,
                thresholds=thresholds,
            )
        )
    if "layout_dino_similarity" in requested_metrics:
        results.append(
            compute_layout_dino_similarity(
                reference_image=reference_image,
                candidate_image=output_image,
                thresholds=thresholds,
            )
        )
    if "layout_dino_object_similarity" in requested_metrics:
        results.append(
            compute_layout_dino_object_similarity(
                reference_image=reference_image,
                candidate_image=output_image,
                source_payload=source_payload,
                thresholds=thresholds,
            )
        )
    if "layout_dino_patch_similarity" in requested_metrics:
        results.append(
            compute_layout_dino_patch_similarity(
                reference_image=reference_image,
                candidate_image=output_image,
                thresholds=thresholds,
            )
        )
    if "layout_edge_structure" in requested_metrics:
        results.append(
            compute_layout_edge_structure(
                reference_image=reference_image,
                candidate_image=output_image,
                thresholds=thresholds,
            )
        )
    if "layout_depth_structure" in requested_metrics:
        results.append(
            compute_layout_depth_structure(
                reference_image=reference_image,
                candidate_image=output_image,
                thresholds=thresholds,
            )
        )
    if "detector_layout" in requested_metrics:
        results.append(
            compute_detector_layout(
                candidate_image=output_image,
                source_payload=source_payload,
                thresholds=thresholds,
            )
        )
    if "detector_spatial_relations" in requested_metrics:
        results.append(
            compute_detector_spatial_relations(
                candidate_image=output_image,
                source_payload=source_payload,
                thresholds=thresholds,
            )
        )
    return results


def compute_layout_gate_score(
    *,
    metrics: Sequence[MetricResult],
    thresholds: Mapping[str, float],
) -> MetricResult:
    layout_metric = _metric_by_name(metrics, "layout_fidelity")
    if layout_metric is None:
        layout_metric = _metric_by_name(metrics, "render_layout_fidelity")
    if layout_metric is None:
        return MetricResult(
            name="layout_gate_score",
            status="failed",
            details={"reason": "layout_fidelity_metric_missing"},
        )

    component_details: dict[str, JSONValue] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, weight in _LAYOUT_GATE_WEIGHTS.items():
        metric = (
            layout_metric
            if name == "layout_fidelity"
            else _metric_by_name(metrics, name)
        )
        component_key = layout_metric.name if name == "layout_fidelity" else name
        if metric is None:
            component_details[name] = {
                "metric": component_key,
                "included": False,
                "reason": "metric_missing",
                "weight": weight,
            }
            continue
        value = _metric_score_value(metric)
        if value is None:
            component_details[name] = {
                "metric": metric.name,
                "included": False,
                "reason": "metric_value_missing",
                "status": metric.status,
                "weight": weight,
            }
            continue
        if metric.status == "skipped":
            component_details[name] = {
                "metric": metric.name,
                "included": False,
                "reason": "metric_skipped",
                "status": metric.status,
                "value": round(value, 6),
                "weight": weight,
            }
            continue
        weighted_sum += weight * value
        weight_sum += weight
        component_details[name] = {
            "metric": metric.name,
            "included": True,
            "status": metric.status,
            "value": round(value, 6),
            "weight": weight,
        }

    if weight_sum <= 0.0:
        return MetricResult(
            name="layout_gate_score",
            status="failed",
            details={
                "reason": "no_scored_components",
                "components": component_details,
            },
        )

    raw_score = _clamp01(weighted_sum / weight_sum)
    score_cap = _layout_metric_score_cap(layout_metric)
    effective_score = raw_score if score_cap is None else min(raw_score, score_cap)
    threshold = thresholds.get("min_layout_gate_score", 0.65)
    passed = layout_metric.status == "passed" and effective_score >= threshold
    return MetricResult(
        name="layout_gate_score",
        status="passed" if passed else "failed",
        value=effective_score,
        threshold=threshold,
        direction="min",
        details={
            "layout_metric": layout_metric.name,
            "layout_status": layout_metric.status,
            "raw_score": round(raw_score, 6),
            "effective_score": round(effective_score, 6),
            "score_cap": score_cap,
            "components": component_details,
        },
    )


def compute_structural_layout_score(
    *,
    metrics: Sequence[MetricResult],
    thresholds: Mapping[str, float],
) -> MetricResult:
    component_details: dict[str, JSONValue] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, weight in _STRUCTURAL_LAYOUT_WEIGHTS.items():
        metric = _metric_by_name(metrics, name)
        if metric is None:
            component_details[name] = {
                "included": False,
                "reason": "metric_missing",
                "weight": weight,
            }
            continue
        value = _metric_score_value(metric)
        if value is None:
            component_details[name] = {
                "included": False,
                "reason": "metric_value_missing",
                "status": metric.status,
                "weight": weight,
            }
            continue
        if metric.status == "skipped":
            component_details[name] = {
                "included": False,
                "reason": "metric_skipped",
                "status": metric.status,
                "value": round(value, 6),
                "weight": weight,
            }
            continue
        weighted_sum += weight * value
        weight_sum += weight
        component_details[name] = {
            "included": True,
            "metric": metric.name,
            "status": metric.status,
            "value": round(value, 6),
            "weight": weight,
        }

    if weight_sum <= 0.0:
        return MetricResult(
            name="structural_layout_score",
            status="skipped",
            details={
                "reason": "no_scored_structural_components",
                "components": component_details,
            },
        )

    score = _clamp01(weighted_sum / weight_sum)
    threshold = thresholds.get("min_structural_layout_score", 0.58)
    return MetricResult(
        name="structural_layout_score",
        status="passed" if score >= threshold else "failed",
        value=score,
        threshold=threshold,
        direction="min",
        details={
            "score_role": "primary_layout_ranking_metric",
            "notes": (
                "LLM layout_fidelity/layout_gate_score are diagnostic only for "
                "ranking; this score emphasizes structure-preserving local metrics."
            ),
            "components": component_details,
            "weight_sum": round(weight_sum, 6),
        },
    )


def compute_layout_lpips(
    *,
    reference_image: Path | None,
    candidate_image: Path,
    thresholds: Mapping[str, float],
) -> MetricResult:
    if reference_image is None:
        return _skipped("layout_lpips", "reference_image_missing")
    try:
        reference, candidate = _aligned_rgb_images(reference_image, candidate_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="layout_lpips",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    try:
        loss_fn, device, net = _lpips_model()
        ref_tensor = _torch_image_tensor(reference, device=device)
        cand_tensor = _torch_image_tensor(candidate, device=device)
        torch = _torch_module()
        with torch.inference_mode():
            value = float(loss_fn(ref_tensor, cand_tensor).item())
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "layout_lpips", f"lpips_unavailable: {type(exc).__name__}: {exc}"
        )

    threshold = thresholds.get("max_layout_lpips")
    return MetricResult(
        name="layout_lpips",
        status=_status_for_threshold(
            value=value,
            threshold=threshold,
            direction="max",
        ),
        value=value,
        threshold=threshold,
        direction="max",
        details={
            "reference_image": str(reference_image),
            "candidate_image": str(candidate_image),
            "model": f"lpips-{net}",
            "device": device,
        },
    )


def compute_layout_dino_similarity(
    *,
    reference_image: Path | None,
    candidate_image: Path,
    thresholds: Mapping[str, float],
) -> MetricResult:
    if reference_image is None:
        return _skipped("layout_dino_similarity", "reference_image_missing")
    try:
        reference, candidate = _aligned_rgb_images(reference_image, candidate_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="layout_dino_similarity",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    try:
        embeddings = _dino_embeddings_for_images([reference, candidate])
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "layout_dino_similarity",
            f"dino_unavailable: {type(exc).__name__}: {exc}",
        )
    similarity = _cosine_similarity(embeddings[0], embeddings[1])
    threshold = thresholds.get("min_layout_dino_similarity")
    return MetricResult(
        name="layout_dino_similarity",
        status=_status_for_threshold(
            value=similarity,
            threshold=threshold,
            direction="min",
        ),
        value=similarity,
        threshold=threshold,
        direction="min",
        details={
            "reference_image": str(reference_image),
            "candidate_image": str(candidate_image),
            "model": _dino_model_name(),
            "device": _metric_device_name(),
        },
    )


def compute_layout_dino_object_similarity(
    *,
    reference_image: Path | None,
    candidate_image: Path,
    source_payload: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> MetricResult:
    if reference_image is None:
        return _skipped("layout_dino_object_similarity", "reference_image_missing")
    try:
        reference, candidate = _aligned_rgb_images(reference_image, candidate_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="layout_dino_object_similarity",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    max_objects = int(
        thresholds.get("layout_dino_object_max_objects", _OBJECT_METRIC_MAX_OBJECTS)
    )
    objects = _visible_object_specs(source_payload, image_size=reference.size)[
        :max_objects
    ]
    if not objects:
        return _skipped("layout_dino_object_similarity", "no_visible_object_bboxes")

    crops: list[Image.Image] = []
    for item in objects:
        crop_box = _padded_crop_box(item.bbox_px, reference.size)
        crops.extend((reference.crop(crop_box), candidate.crop(crop_box)))
    try:
        embeddings = _dino_embeddings_for_images(crops)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "layout_dino_object_similarity",
            f"dino_unavailable: {type(exc).__name__}: {exc}",
        )

    object_scores: list[dict[str, JSONValue]] = []
    similarities: list[float] = []
    for index, item in enumerate(objects):
        ref_embedding = embeddings[index * 2]
        candidate_embedding = embeddings[index * 2 + 1]
        similarity = _cosine_similarity(ref_embedding, candidate_embedding)
        similarities.append(similarity)
        object_scores.append(
            {
                "object_id": item.object_id,
                "object_type": item.object_type,
                "bbox_norm": dict(item.bbox_norm),
                "similarity": round(similarity, 6),
            }
        )
    value = float(np.mean(np.asarray(similarities, dtype=np.float32)))
    threshold = thresholds.get("min_layout_dino_object_similarity")
    return MetricResult(
        name="layout_dino_object_similarity",
        status=_status_for_threshold(value=value, threshold=threshold, direction="min"),
        value=value,
        threshold=threshold,
        direction="min",
        details={
            "reference_image": str(reference_image),
            "candidate_image": str(candidate_image),
            "model": _dino_model_name(),
            "device": _metric_device_name(),
            "object_count": len(objects),
            "objects": object_scores,
        },
    )


def compute_layout_dino_patch_similarity(
    *,
    reference_image: Path | None,
    candidate_image: Path,
    thresholds: Mapping[str, float],
) -> MetricResult:
    if reference_image is None:
        return _skipped("layout_dino_patch_similarity", "reference_image_missing")
    try:
        reference, candidate = _aligned_rgb_images(reference_image, candidate_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="layout_dino_patch_similarity",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    try:
        embeddings = _dino_patch_embeddings_for_images([reference, candidate])
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "layout_dino_patch_similarity",
            f"dino_unavailable: {type(exc).__name__}: {exc}",
        )
    if embeddings.shape[0] < 2:
        return _skipped("layout_dino_patch_similarity", "patch_embedding_missing")
    patch_count = min(embeddings[0].shape[0], embeddings[1].shape[0])
    if patch_count <= 0:
        return _skipped("layout_dino_patch_similarity", "patch_tokens_missing")
    similarities = np.sum(
        embeddings[0][:patch_count] * embeddings[1][:patch_count],
        axis=1,
    )
    value = float(np.mean(np.clip(similarities, 0.0, 1.0)))
    threshold = thresholds.get("min_layout_dino_patch_similarity")
    return MetricResult(
        name="layout_dino_patch_similarity",
        status=_status_for_threshold(value=value, threshold=threshold, direction="min"),
        value=value,
        threshold=threshold,
        direction="min",
        details={
            "reference_image": str(reference_image),
            "candidate_image": str(candidate_image),
            "model": _dino_model_name(),
            "device": _metric_device_name(),
            "patch_count": patch_count,
        },
    )


def compute_layout_edge_structure(
    *,
    reference_image: Path | None,
    candidate_image: Path,
    thresholds: Mapping[str, float],
) -> MetricResult:
    if reference_image is None:
        return _skipped("layout_edge_structure", "reference_image_missing")
    try:
        reference, candidate = _aligned_rgb_images(reference_image, candidate_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="layout_edge_structure",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    percentile = thresholds.get("layout_edge_percentile", _EDGE_PERCENTILE)
    tolerance_px = int(
        thresholds.get("layout_edge_match_radius_px", _EDGE_DILATION_PIXELS)
    )
    reference_edges = _image_edge_mask(reference, percentile=percentile)
    candidate_edges = _image_edge_mask(candidate, percentile=percentile)
    stats = _edge_match_stats(
        reference_edges,
        candidate_edges,
        tolerance_px=max(0, tolerance_px),
    )
    value = stats["f1"]
    threshold = thresholds.get("min_layout_edge_structure_score")
    return MetricResult(
        name="layout_edge_structure",
        status=_status_for_threshold(value=value, threshold=threshold, direction="min"),
        value=value,
        threshold=threshold,
        direction="min",
        details={
            "reference_image": str(reference_image),
            "candidate_image": str(candidate_image),
            "edge_percentile": percentile,
            "match_radius_px": max(0, tolerance_px),
            **{key: round(item, 6) for key, item in stats.items()},
        },
    )


def compute_layout_depth_structure(
    *,
    reference_image: Path | None,
    candidate_image: Path,
    thresholds: Mapping[str, float],
) -> MetricResult:
    if reference_image is None:
        return _skipped("layout_depth_structure", "reference_image_missing")
    try:
        reference, candidate = _aligned_rgb_images(reference_image, candidate_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="layout_depth_structure",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    try:
        depth_maps = _depth_maps_for_images([reference, candidate])
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "layout_depth_structure",
            f"depth_model_unavailable: {type(exc).__name__}: {exc}",
        )
    if depth_maps.shape[0] < 2:
        return _skipped("layout_depth_structure", "depth_map_missing")
    reference_depth = _normalize_depth_map(depth_maps[0])
    candidate_depth = _normalize_depth_map(depth_maps[1])
    if reference_depth.shape != candidate_depth.shape:
        candidate_depth = _resize_float_array(candidate_depth, reference_depth.shape)
    depth_similarity = _clamp01(
        1.0 - float(np.mean(np.abs(reference_depth - candidate_depth)))
    )
    percentile = thresholds.get("layout_depth_edge_percentile", _EDGE_PERCENTILE)
    tolerance_px = int(
        thresholds.get("layout_depth_edge_match_radius_px", _EDGE_DILATION_PIXELS)
    )
    reference_edges = _edge_mask_from_float_array(
        reference_depth,
        percentile=percentile,
    )
    candidate_edges = _edge_mask_from_float_array(
        candidate_depth,
        percentile=percentile,
    )
    edge_stats = _edge_match_stats(
        reference_edges,
        candidate_edges,
        tolerance_px=max(0, tolerance_px),
    )
    depth_edge_similarity = edge_stats["f1"]
    value = _clamp01(0.4 * depth_similarity + 0.6 * depth_edge_similarity)
    threshold = thresholds.get("min_layout_depth_structure_score")
    return MetricResult(
        name="layout_depth_structure",
        status=_status_for_threshold(value=value, threshold=threshold, direction="min"),
        value=value,
        threshold=threshold,
        direction="min",
        details={
            "reference_image": str(reference_image),
            "candidate_image": str(candidate_image),
            "model": _depth_model_name(),
            "device": _metric_device_name(),
            "depth_similarity": round(depth_similarity, 6),
            "depth_edge_similarity": round(depth_edge_similarity, 6),
            "edge_percentile": percentile,
            "match_radius_px": max(0, tolerance_px),
            "edge_stats": {key: round(item, 6) for key, item in edge_stats.items()},
        },
    )


def compute_detector_layout(
    *,
    candidate_image: Path,
    source_payload: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> MetricResult:
    try:
        image = _open_rgba(candidate_image).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="detector_layout",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    expected = [
        item
        for item in _visible_object_specs(source_payload, image_size=image.size)
        if item.detector_label is not None
    ]
    if not expected:
        return _skipped("detector_layout", "no_detector_supported_visible_objects")
    try:
        detections = _detect_objects(candidate_image)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "detector_layout",
            f"detector_unavailable: {type(exc).__name__}: {exc}",
        )
    match_stats = _match_expected_detections(expected, detections)
    match_ratio = match_stats["match_ratio"]
    threshold = thresholds.get("min_detector_layout_match_ratio")
    status = _status_for_threshold(
        value=match_ratio,
        threshold=threshold,
        direction="min",
    )
    return MetricResult(
        name="detector_layout",
        status=status,
        value=match_ratio,
        threshold=threshold,
        direction="min",
        details={
            "candidate_image": str(candidate_image),
            "model": _yolo_model_name(),
            **match_stats,
        },
    )


def compute_detector_spatial_relations(
    *,
    candidate_image: Path,
    source_payload: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> MetricResult:
    try:
        image = _open_rgba(candidate_image).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="detector_spatial_relations",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )
    expected = [
        item
        for item in _visible_object_specs(source_payload, image_size=image.size)
        if item.detector_label is not None
    ]
    if len(expected) < 2:
        return _skipped(
            "detector_spatial_relations",
            "not_enough_detector_supported_visible_objects",
        )
    try:
        detections = _detect_objects(candidate_image)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return _skipped(
            "detector_spatial_relations",
            f"detector_unavailable: {type(exc).__name__}: {exc}",
        )
    match_stats = _match_expected_detections(expected, detections)
    matched = cast(dict[str, Mapping[str, object]], match_stats["matched_objects"])
    relation_stats = _spatial_relation_stats(expected, matched)
    value = relation_stats["relation_match_ratio"]
    threshold = thresholds.get("min_detector_spatial_relation_match_ratio")
    return MetricResult(
        name="detector_spatial_relations",
        status=_status_for_threshold(value=value, threshold=threshold, direction="min"),
        value=value,
        threshold=threshold,
        direction="min",
        details={
            "candidate_image": str(candidate_image),
            "model": _yolo_model_name(),
            **relation_stats,
        },
    )


def compute_pair_metrics(
    *,
    prefix: str,
    reference_path: Path,
    candidate_path: Path,
    mask_path: Path | None,
    selected_region: str,
    output_dir: Path,
    metrics: set[str],
    thresholds: Mapping[str, float],
    visual_delta_threshold: int,
    align_mode: AlignMode,
    case_id: str,
    variant_id: str,
) -> list[MetricResult]:
    try:
        pair = _prepare_pair(
            reference_path=reference_path,
            candidate_path=candidate_path,
            align_mode=align_mode,
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        return [
            MetricResult(
                name=f"{prefix}_alignment",
                status="failed",
                details={"reason": str(exc)},
            )
        ]

    selected_mask = _load_selected_mask(
        mask_path=mask_path,
        size=pair.reference.size,
        selected_region=selected_region,
    )
    reference_rgb = _rgb_array(pair.reference)
    candidate_rgb = _rgb_array(pair.candidate)
    results: list[MetricResult] = []

    if "visual_diff" in metrics or "visual" in metrics:
        results.append(
            compute_visual_diff(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                output_dir=output_dir,
                case_id=case_id,
                variant_id=variant_id,
                thresholds=thresholds,
                visual_delta_threshold=visual_delta_threshold,
                extra_details=pair.details,
            )
        )

    if "psnr" in metrics:
        results.append(
            compute_psnr(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                thresholds=thresholds,
                extra_details=pair.details,
            )
        )

    if "ssim" in metrics:
        results.append(
            compute_ssim(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                thresholds=thresholds,
                extra_details=pair.details,
            )
        )

    if "nrmse" in metrics:
        results.append(
            compute_nrmse(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                thresholds=thresholds,
                extra_details=pair.details,
            )
        )

    return results


def compute_visual_diff(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    output_dir: Path,
    case_id: str,
    variant_id: str,
    thresholds: Mapping[str, float],
    visual_delta_threshold: int,
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    selected = _selected_mask_or_all(selected_mask, reference_rgb.shape[:2])
    if not bool(np.any(selected)):
        return _skipped(f"{prefix}_visual_diff", "selected_region_is_empty")

    diff = np.abs(reference_rgb.astype(np.int16) - candidate_rgb.astype(np.int16))
    channel_max = np.max(diff, axis=2)
    selected_channel_max = channel_max[selected]
    selected_diff = diff[selected]
    diff_pixel_ratio = float(np.mean(selected_channel_max > visual_delta_threshold))
    mean_abs_error = float(np.mean(selected_diff))
    max_channel_delta = int(np.max(selected_channel_max))
    diff_path = _write_diff_image(
        diff=diff,
        selected_mask=selected,
        output_dir=output_dir,
        case_id=case_id,
        variant_id=variant_id,
        prefix=prefix,
    )
    max_ratio = _threshold(thresholds, prefix, "max_diff_pixel_ratio")
    max_mean_abs_error = _threshold(thresholds, prefix, "max_mean_abs_error")
    passed = True
    if max_ratio is not None:
        passed = passed and diff_pixel_ratio <= max_ratio
    if max_mean_abs_error is not None:
        passed = passed and mean_abs_error <= max_mean_abs_error
    status: MetricStatus = "passed" if passed else "failed"
    if max_ratio is None and max_mean_abs_error is None:
        status = "warning"
    details = {
        **dict(extra_details),
        "diff_pixel_ratio": diff_pixel_ratio,
        "mean_abs_error": mean_abs_error,
        "max_channel_delta": max_channel_delta,
        "visual_delta_threshold": visual_delta_threshold,
        "diff_image": str(diff_path),
    }
    if max_mean_abs_error is not None:
        details["max_mean_abs_error"] = max_mean_abs_error
    return MetricResult(
        name=f"{prefix}_visual_diff",
        status=status,
        value=diff_pixel_ratio,
        threshold=max_ratio,
        direction="max",
        details=details,
    )


def compute_psnr(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    thresholds: Mapping[str, float],
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    selected_reference, selected_candidate = _selected_pixels(
        reference_rgb,
        candidate_rgb,
        selected_mask,
    )
    if selected_reference.size == 0:
        return _skipped(f"{prefix}_psnr", "selected_region_is_empty")
    mse = float(np.mean((selected_reference - selected_candidate) ** 2))
    perfect_match = mse == 0.0
    value = 100.0 if perfect_match else 20.0 * math.log10(255.0 / math.sqrt(mse))
    threshold = _threshold(thresholds, prefix, "min_psnr")
    status = _status_for_threshold(value=value, threshold=threshold, direction="min")
    return MetricResult(
        name=f"{prefix}_psnr",
        status=status,
        value=value,
        threshold=threshold,
        direction="min",
        details={**dict(extra_details), "mse": mse, "perfect_match": perfect_match},
    )


def compute_nrmse(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    thresholds: Mapping[str, float],
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    selected_reference, selected_candidate = _selected_pixels(
        reference_rgb,
        candidate_rgb,
        selected_mask,
    )
    if selected_reference.size == 0:
        return _skipped(f"{prefix}_nrmse", "selected_region_is_empty")
    rmse = float(np.sqrt(np.mean((selected_reference - selected_candidate) ** 2)))
    value = rmse / 255.0
    threshold = _threshold(thresholds, prefix, "max_nrmse")
    status = _status_for_threshold(value=value, threshold=threshold, direction="max")
    return MetricResult(
        name=f"{prefix}_nrmse",
        status=status,
        value=value,
        threshold=threshold,
        direction="max",
        details={**dict(extra_details), "rmse": rmse, "normalization": "255"},
    )


def compute_ssim(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    thresholds: Mapping[str, float],
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return _skipped(f"{prefix}_ssim", "scikit-image is not installed")

    min_side = min(reference_rgb.shape[0], reference_rgb.shape[1])
    if min_side < 3:
        return _skipped(f"{prefix}_ssim", "image_too_small")
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if selected_mask is not None:
        ignored = np.logical_not(selected_mask)
        candidate_for_metric = candidate_rgb.copy()
        candidate_for_metric[ignored] = reference_rgb[ignored]
    else:
        candidate_for_metric = candidate_rgb
    value = float(
        structural_similarity(
            reference_rgb,
            candidate_for_metric,
            channel_axis=2,
            data_range=255,
            win_size=win_size,
        )
    )
    threshold = _threshold(thresholds, prefix, "min_ssim")
    status = _status_for_threshold(value=value, threshold=threshold, direction="min")
    return MetricResult(
        name=f"{prefix}_ssim",
        status=status,
        value=value,
        threshold=threshold,
        direction="min",
        details={**dict(extra_details), "win_size": win_size},
    )


def summarize_human_ratings(
    *,
    ratings_csv: Path,
    artifact_fail_threshold: float = 3.0,
) -> dict[str, JSONValue]:
    groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with ratings_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            variant_id = (row.get("variant") or row.get("variant_id") or "").strip()
            if not case_id or not variant_id:
                continue
            groups[(case_id, variant_id)].append(
                {
                    "score_instruction": _optional_float(row.get("score_instruction")),
                    "score_quality": _optional_float(row.get("score_quality")),
                    "score_preservation": _optional_float(
                        row.get("score_preservation")
                    ),
                    "artifact_severity": _optional_float(row.get("artifact_severity")),
                }
            )
    output: dict[str, JSONValue] = {}
    for (case_id, variant_id), rows in groups.items():
        key = f"{case_id}/{variant_id}"
        output[key] = {
            "case_id": case_id,
            "variant_id": variant_id,
            "rater_count": len(rows),
            "mean_instruction": _mean(row["score_instruction"] for row in rows),
            "mean_quality": _mean(row["score_quality"] for row in rows),
            "mean_preservation": _mean(row["score_preservation"] for row in rows),
            "mean_artifact_severity": _mean(row["artifact_severity"] for row in rows),
            "artifact_fail_rate": _rate(
                row["artifact_severity"] >= artifact_fail_threshold for row in rows
            ),
        }
    return output


def _prepare_pair(
    *,
    reference_path: Path,
    candidate_path: Path,
    align_mode: AlignMode,
) -> PairImages:
    reference = _open_rgba(reference_path).convert("RGB")
    candidate = _open_rgba(candidate_path).convert("RGB")
    details: dict[str, JSONValue] = {
        "reference_image": str(reference_path),
        "candidate_image": str(candidate_path),
        "reference_width": reference.width,
        "reference_height": reference.height,
        "candidate_width": candidate.width,
        "candidate_height": candidate.height,
        "candidate_resized": False,
    }
    if reference.size != candidate.size:
        if align_mode == "strict":
            raise ValueError(
                f"Image dimensions differ: reference={reference.size}, "
                f"candidate={candidate.size}."
            )
        candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
        details["candidate_resized"] = True
        details["aligned_width"] = reference.width
        details["aligned_height"] = reference.height
    return PairImages(reference=reference, candidate=candidate, details=details)


def _local_layout_reference_image(
    case: CaseConfig,
    provider_result: ProviderResult | None,
) -> Path | None:
    if (
        case.edit_operations
        and provider_result is not None
        and provider_result.render_image is not None
    ):
        return provider_result.render_image
    return _layout_source_image(case, provider_result)


def _layout_source_image(
    case: CaseConfig,
    provider_result: ProviderResult | None,
) -> Path | None:
    if provider_result is not None and provider_result.source_image is not None:
        return provider_result.source_image
    return case.snapshot_image or case.source_image


def _aligned_rgb_images(
    reference_path: Path,
    candidate_path: Path,
) -> tuple[Image.Image, Image.Image]:
    reference = _open_rgba(reference_path).convert("RGB")
    candidate = _open_rgba(candidate_path).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
    return reference, candidate


def _image_edge_mask(
    image: Image.Image,
    *,
    percentile: float,
) -> BoolMask:
    return _edge_mask_from_float_array(_grayscale_float(image), percentile=percentile)


def _grayscale_float(image: Image.Image) -> FloatArray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return cast(
        FloatArray,
        rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114,
    )


def _edge_mask_from_float_array(
    array: FloatArray,
    *,
    percentile: float,
) -> BoolMask:
    if array.size == 0:
        return cast(BoolMask, np.zeros(array.shape, dtype=np.bool_))
    blurred = _box_blur_3x3(array.astype(np.float32, copy=False))
    magnitude = _sobel_magnitude(blurred)
    positive = magnitude[magnitude > 1e-6]
    if positive.size == 0:
        return cast(BoolMask, np.zeros(magnitude.shape, dtype=np.bool_))
    clipped_percentile = min(99.5, max(0.0, percentile))
    cutoff = float(np.percentile(positive, clipped_percentile))
    return cast(BoolMask, magnitude >= cutoff)


def _box_blur_3x3(array: FloatArray) -> FloatArray:
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    blurred = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return cast(FloatArray, blurred.astype(np.float32, copy=False))


def _sobel_magnitude(array: FloatArray) -> FloatArray:
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    grad_x = (
        -padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    grad_y = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    max_value = float(np.max(magnitude)) if magnitude.size else 0.0
    if max_value <= 0.0:
        return cast(FloatArray, magnitude.astype(np.float32, copy=False))
    return cast(FloatArray, (magnitude / max_value).astype(np.float32, copy=False))


def _edge_match_stats(
    reference_edges: BoolMask,
    candidate_edges: BoolMask,
    *,
    tolerance_px: int,
) -> dict[str, float]:
    reference_count = float(np.count_nonzero(reference_edges))
    candidate_count = float(np.count_nonzero(candidate_edges))
    if reference_count == 0.0 and candidate_count == 0.0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "reference_edge_ratio": 0.0,
            "candidate_edge_ratio": 0.0,
        }
    if reference_count == 0.0 or candidate_count == 0.0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "reference_edge_ratio": reference_count / max(1, reference_edges.size),
            "candidate_edge_ratio": candidate_count / max(1, candidate_edges.size),
        }
    reference_match_mask = _dilate_mask(reference_edges, tolerance_px)
    candidate_match_mask = _dilate_mask(candidate_edges, tolerance_px)
    matched_candidate = float(
        np.count_nonzero(np.logical_and(candidate_edges, reference_match_mask))
    )
    matched_reference = float(
        np.count_nonzero(np.logical_and(reference_edges, candidate_match_mask))
    )
    precision = matched_candidate / candidate_count
    recall = matched_reference / reference_count
    f1 = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    tolerant_iou = 0.0 if f1 <= 0.0 else f1 / (2.0 - f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tolerant_iou,
        "reference_edge_ratio": reference_count / max(1, reference_edges.size),
        "candidate_edge_ratio": candidate_count / max(1, candidate_edges.size),
    }


def _dilate_mask(mask: BoolMask, radius: int) -> BoolMask:
    if radius <= 0:
        return mask
    height, width = mask.shape
    padded = np.pad(
        mask,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=False,
    )
    dilated = np.zeros_like(mask, dtype=np.bool_)
    for offset_y in range(radius * 2 + 1):
        for offset_x in range(radius * 2 + 1):
            dilated |= padded[offset_y : offset_y + height, offset_x : offset_x + width]
    return cast(BoolMask, dilated)


def _normalize_depth_map(depth_map: FloatArray) -> FloatArray:
    clean = depth_map.astype(np.float32, copy=False)
    finite = clean[np.isfinite(clean)]
    if finite.size == 0:
        return cast(FloatArray, np.zeros(clean.shape, dtype=np.float32))
    low = float(np.percentile(finite, 5.0))
    high = float(np.percentile(finite, 95.0))
    if high <= low:
        return cast(FloatArray, np.zeros(clean.shape, dtype=np.float32))
    normalized = np.clip((clean - low) / (high - low), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return cast(FloatArray, normalized.astype(np.float32, copy=False))


def _resize_float_array(
    array: FloatArray,
    shape: tuple[int, int],
) -> FloatArray:
    target_height, target_width = shape
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    resized = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return cast(FloatArray, np.asarray(resized, dtype=np.float32) / 255.0)


def _layout_source_payload(
    case: CaseConfig,
    provider_result: ProviderResult | None,
) -> Mapping[str, object]:
    path = (
        provider_result.source_snapshot_payload
        if provider_result is not None
        and provider_result.source_snapshot_payload is not None
        else case.snapshot_payload
    )
    if path is None or not path.is_file():
        return {}
    try:
        return _read_json_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _provider_metadata_runtime(path: Path | None) -> float | None:
    metadata = _provider_metadata(path)
    if not metadata:
        return None
    runtime_s = _optional_number(metadata.get("provider_runtime_s"))
    if runtime_s is not None:
        return runtime_s
    elapsed_ms_values = [
        _optional_number(item.get("elapsed_ms"))
        for item in _provider_render_sections(metadata)
        if isinstance(item, Mapping)
    ]
    clean = [value for value in elapsed_ms_values if value is not None]
    if clean:
        return sum(clean) / 1000.0
    return None


def _provider_token_usage(path: Path | None) -> dict[str, JSONValue]:
    metadata = _provider_metadata(path)
    source_usages: list[JSONValue] = []
    missing_usage_sources: list[JSONValue] = []
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cached_input_tokens = 0

    def include_usage(source: str, usage: Mapping[str, object] | None) -> None:
        nonlocal input_tokens, output_tokens, total_tokens, cached_input_tokens
        if usage is None:
            missing_usage_sources.append(source)
            return
        parsed = _token_usage_counts(usage)
        input_tokens += parsed["input_tokens"]
        output_tokens += parsed["output_tokens"]
        total_tokens += parsed["total_tokens"]
        cached_input_tokens += parsed["cached_input_tokens"]
        source_usages.append(
            {
                "source": source,
                "input_tokens": parsed["input_tokens"],
                "output_tokens": parsed["output_tokens"],
                "total_tokens": parsed["total_tokens"],
                "cached_input_tokens": parsed["cached_input_tokens"],
                "raw_usage": _json_safe(usage),
            }
        )

    if not metadata:
        return {
            "metadata_path": str(path) if path is not None else None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "source_usages": [],
            "missing_usage_sources": ["metadata_missing"],
        }

    visual_preflight = _expect_optional_mapping(metadata.get("visual_preflight"))
    if visual_preflight is not None:
        include_usage(
            "visual_preflight",
            _expect_optional_mapping(visual_preflight.get("usage")),
        )

    for source_name, section in _provider_render_sections_by_name(metadata):
        include_usage(source_name, _render_section_usage(section))

    return {
        "metadata_path": str(path) if path is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "source_usages": source_usages,
        "missing_usage_sources": missing_usage_sources,
    }


def _provider_metadata(path: Path | None) -> Mapping[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        return _read_json_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _provider_semantic_layout_locks(path: Path | None) -> tuple[str, ...]:
    metadata = _provider_metadata(path)
    if not metadata:
        return ()
    guide = metadata.get("edge_silhouette_guide")
    if not isinstance(guide, Mapping):
        return ()
    raw_locks = guide.get("semantic_room_shell_locks")
    if not isinstance(raw_locks, list):
        return ()
    locks = [
        item.strip() for item in raw_locks if isinstance(item, str) and item.strip()
    ]
    return tuple(locks[:12])


def _provider_render_sections(
    metadata: Mapping[str, object],
) -> list[Mapping[str, object]]:
    return [section for _, section in _provider_render_sections_by_name(metadata)]


def _provider_render_sections_by_name(
    metadata: Mapping[str, object],
) -> list[tuple[str, Mapping[str, object]]]:
    sections: list[tuple[str, Mapping[str, object]]] = []
    pass_results = _expect_optional_mapping(metadata.get("pass_results"))
    if pass_results:
        for pass_name in ("pass1", "pass2"):
            section = _expect_optional_mapping(pass_results.get(pass_name))
            if section is not None:
                sections.append((f"pass_results.{pass_name}", section))
        edit_section = _expect_optional_mapping(metadata.get("edit_result"))
        if edit_section is not None:
            sections.append(("edit_result", edit_section))
        return sections

    for source_name in ("render_result", "edit_result"):
        section = _expect_optional_mapping(metadata.get(source_name))
        if section is not None:
            sections.append((source_name, section))
    return sections


def _render_section_usage(section: Mapping[str, object]) -> Mapping[str, object] | None:
    usage = _expect_optional_mapping(section.get("usage"))
    if usage is not None:
        return usage
    image = _expect_optional_mapping(section.get("image"))
    if image is None:
        return None
    return _expect_optional_mapping(image.get("usage"))


def _token_usage_counts(usage: Mapping[str, object]) -> dict[str, int]:
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    cached_input_tokens = 0
    input_details = _expect_optional_mapping(
        usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    )
    if input_details is not None:
        cached_input_tokens = _usage_int(input_details, "cached_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
    }


def _usage_int(usage: Mapping[str, object], *keys: str) -> int:
    for key in keys:
        value = _optional_number(usage.get(key))
        if value is not None:
            return max(0, int(round(value)))
    return 0


def _visible_object_ids(source_payload: Mapping[str, object]) -> list[JSONValue]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    ids: list[JSONValue] = []
    for item in visible_objects:
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def _visible_object_details(source_payload: Mapping[str, object]) -> list[JSONValue]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    canvas = source_payload.get("canvas")
    canvas_payload = canvas if isinstance(canvas, Mapping) else {}
    canvas_width = _optional_float(canvas_payload.get("widthPx"))
    canvas_height = _optional_float(canvas_payload.get("heightPx"))
    details: list[JSONValue] = []
    for item in visible_objects:
        if not isinstance(item, Mapping):
            continue
        bbox = item.get("screenBboxPx")
        bbox_payload = bbox if isinstance(bbox, Mapping) else {}
        details.append(
            {
                "id": str(item.get("id") or ""),
                "type": str(item.get("type") or ""),
                "screen_bbox_px": _json_safe(bbox),
                "dimensions_mm": _json_safe(item.get("dimensionsMm")),
                "distance_to_camera_m": _json_safe(item.get("distanceToCameraM")),
                "clipped": _bbox_touches_canvas(
                    bbox_payload,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                ),
            }
        )
    return details


def _visible_object_specs(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> list[VisibleObjectSpec]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    width, height = image_size
    specs: list[VisibleObjectSpec] = []
    for item in visible_objects:
        if not isinstance(item, Mapping):
            continue
        object_id = _clean_text(item.get("id"))
        if object_id is None:
            continue
        bbox = _screen_bbox_px(item.get("screenBboxPx"))
        if bbox is None:
            continue
        bbox_width = bbox["maxX"] - bbox["minX"]
        bbox_height = bbox["maxY"] - bbox["minY"]
        if (
            bbox_width < _OBJECT_METRIC_MIN_BBOX_PX
            or bbox_height < _OBJECT_METRIC_MIN_BBOX_PX
            or width <= 0
            or height <= 0
        ):
            continue
        object_type = (
            _clean_text(item.get("type"))
            or _clean_text(item.get("label"))
            or _clean_text(item.get("name"))
        )
        bbox_norm = {
            "x": _clamp01(bbox["minX"] / width),
            "y": _clamp01(bbox["minY"] / height),
            "width": _clamp01(bbox_width / width),
            "height": _clamp01(bbox_height / height),
        }
        specs.append(
            VisibleObjectSpec(
                object_id=object_id,
                object_type=object_type,
                detector_label=_detector_label_for_object(object_type or object_id),
                bbox_norm=bbox_norm,
                bbox_px=bbox,
            )
        )
    return specs


def _screen_bbox_px(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    min_x = _optional_number(value.get("minX"))
    min_y = _optional_number(value.get("minY"))
    if min_x is None:
        min_x = _optional_number(value.get("x"))
    if min_y is None:
        min_y = _optional_number(value.get("y"))
    if min_x is None or min_y is None:
        return None
    width = _optional_number(value.get("width"))
    height = _optional_number(value.get("height"))
    max_x = _optional_number(value.get("maxX"))
    max_y = _optional_number(value.get("maxY"))
    if max_x is None and width is not None:
        max_x = min_x + width
    if max_y is None and height is not None:
        max_y = min_y + height
    if max_x is None or max_y is None:
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def _opening_details(source_payload: Mapping[str, object]) -> JSONValue:
    room = source_payload.get("room")
    room_payload = room if isinstance(room, Mapping) else {}
    return _json_safe(room_payload.get("openings"))


def _source_has_window(source_payload: Mapping[str, object]) -> bool:
    room = source_payload.get("room")
    room_payload = room if isinstance(room, Mapping) else {}
    openings = room_payload.get("openings")
    openings_payload = openings if isinstance(openings, Mapping) else {}
    windows = openings_payload.get("windows")
    return isinstance(windows, list) and bool(windows)


def _scenery_applicable(
    preset_selection: Mapping[str, str],
    source_has_window: bool,
) -> bool:
    scenery = preset_selection.get("scenery")
    return bool(source_has_window and scenery and scenery != "none")


def _preset_selection_details(
    preset_selection: Mapping[str, str],
) -> dict[str, JSONValue]:
    details: dict[str, JSONValue] = {
        "selection": dict(preset_selection),
    }
    try:
        from config import config_file, load_config  # noqa: PLC0415

        presets = load_config(config_file).presets
        groups = {
            "style": presets.styles,
            "lighting": presets.lights,
            "scenery": presets.sceneries,
        }
        for group_name, key in preset_selection.items():
            option = groups.get(group_name, {}).get(key)
            if option is None:
                details[group_name] = {"key": key}
                continue
            details[group_name] = {
                "key": key,
                "label": option.label,
                "prompt_suffix": option.prompt_suffix,
                "reference_image": option.reference_image,
            }
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        details["config_error"] = f"{type(exc).__name__}: {exc}"
    return details


def _bbox_touches_canvas(
    bbox: Mapping[str, object],
    *,
    canvas_width: float,
    canvas_height: float,
) -> bool:
    if not bbox or math.isnan(canvas_width) or math.isnan(canvas_height):
        return False
    margin = 1.0
    return (
        _optional_float(bbox.get("minX")) <= margin
        or _optional_float(bbox.get("minY")) <= margin
        or _optional_float(bbox.get("maxX")) >= canvas_width - margin
        or _optional_float(bbox.get("maxY")) >= canvas_height - margin
    )


def _edit_operation_details(case: CaseConfig) -> list[JSONValue]:
    return [
        {
            "object_id": operation.object_id,
            "operation_type": operation.operation_type,
            "object_name": operation.object_name,
            "prompt": operation.prompt,
            "target_color": operation.target_color,
            "bbox_norm": (
                {
                    "x": operation.bbox_norm.x,
                    "y": operation.bbox_norm.y,
                    "width": operation.bbox_norm.width,
                    "height": operation.bbox_norm.height,
                }
                if operation.bbox_norm is not None
                else None
            ),
        }
        for operation in case.edit_operations
    ]


def _write_fidelity_report(
    output_dir: Path,
    case_id: str,
    variant_id: str,
    filename: str,
    report: Mapping[str, JSONValue],
) -> None:
    path = output_dir / case_id / variant_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _evaluate_fidelity_with_vision(
    *,
    source_image: Path,
    candidate_image: Path,
    source_payload: Mapping[str, object],
    case: CaseConfig,
    variant: VariantConfig,
    deterministic_details: Mapping[str, JSONValue],
    evaluation_kind: FidelityEvaluationKind,
    required_hard_checks: Sequence[str],
    semantic_layout_locks: Sequence[str] = (),
) -> dict[str, JSONValue]:
    api_key = _layout_eval_api_key()
    if not api_key:
        return {
            **dict(deterministic_details),
            "status": "failed",
            "score": 0.0,
            "reason": "layout_fidelity_vision_api_key_missing",
        }
    model = _layout_eval_model()
    prompt = _fidelity_eval_prompt(
        case=case,
        variant=variant,
        source_payload=source_payload,
        evaluation_kind=evaluation_kind,
        required_hard_checks=required_hard_checks,
        semantic_layout_locks=semantic_layout_locks,
    )
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict visual QA evaluator for interior image "
                    "generation. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image_data_url(source_image)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image_data_url(candidate_image)},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    endpoint = (
        os.getenv(_LAYOUT_FIDELITY_BASE_URL_ENV, _DEFAULT_OPENAI_BASE_URL).rstrip("/")
        + "/chat/completions"
    )
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        response.raise_for_status()
        parsed = response.json()
        report = _extract_chat_json(parsed)
    except (httpx.HTTPError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {
            **dict(deterministic_details),
            "status": "failed",
            "score": 0.0,
            "reason": "layout_fidelity_vision_eval_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "model": model,
        }
    return {
        **dict(deterministic_details),
        **report,
        "model": model,
    }


def _evaluate_preset_fidelity_with_vision(
    *,
    source_image: Path | None,
    candidate_image: Path,
    preset_details: Mapping[str, JSONValue],
    deterministic_details: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    api_key = _layout_eval_api_key()
    if not api_key:
        return {
            **dict(deterministic_details),
            "status": "skipped",
            "score": None,
            "reason": "preset_fidelity_vision_api_key_missing",
        }
    model = _layout_eval_model()
    prompt = _preset_fidelity_eval_prompt(
        preset_details=preset_details,
        source_has_window=bool(deterministic_details.get("source_has_window")),
        scenery_applicable=bool(deterministic_details.get("scenery_applicable")),
    )
    user_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    if source_image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_data_url(source_image)},
            }
        )
    user_content.append(
        {
            "type": "image_url",
            "image_url": {"url": _encode_image_data_url(candidate_image)},
        }
    )
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a visual QA evaluator for interior rendering presets. "
                    "Return only JSON."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    endpoint = (
        os.getenv(_LAYOUT_FIDELITY_BASE_URL_ENV, _DEFAULT_OPENAI_BASE_URL).rstrip("/")
        + "/chat/completions"
    )
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        response.raise_for_status()
        parsed = response.json()
        report = _extract_preset_chat_json(parsed)
    except (httpx.HTTPError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {
            **dict(deterministic_details),
            "status": "failed",
            "score": 0.0,
            "reason": "preset_fidelity_vision_eval_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "model": model,
        }
    return {
        **dict(deterministic_details),
        **report,
        "model": model,
    }


def _fidelity_eval_prompt(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    source_payload: Mapping[str, object],
    evaluation_kind: FidelityEvaluationKind,
    required_hard_checks: Sequence[str],
    semantic_layout_locks: Sequence[str] = (),
) -> str:
    visible_objects = json.dumps(
        _visible_object_details(source_payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    openings = json.dumps(
        _opening_details(source_payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    edit_details = json.dumps(
        _edit_operation_details(case),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    preset_details = json.dumps(
        _preset_selection_details(case.preset_selection.to_mapping())
        if case.preset_selection is not None
        else {},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    semantic_lock_text = (
        "\n".join(f"- {item}" for item in semantic_layout_locks[:12])
        if semantic_layout_locks
        else "None"
    )
    edit_rules = (
        "Operation rules: recolor changes only target color/material and preserves "
        "silhouette; replace preserves target footprint, scale, and position; add "
        "creates one object only inside its target bbox; remove deletes only the "
        "target and naturally inpaints its bbox."
        if evaluation_kind == "edit"
        else ""
    )
    if evaluation_kind == "edit":
        task = (
            "Image 1 is the approved photorealistic render and Image 2 is the edit "
            "candidate. The edit candidate may change only the requested target "
            "operation regions/objects. Outside those targets, camera, room shell, "
            "openings, object count, object positions, object scale, materials, "
            "lighting relationship, and visibility must stay unchanged."
        )
    else:
        task = (
            "Image 1 is the exact 3D layout source and Image 2 is the generated "
            "photorealistic candidate. The candidate may change only realism, "
            "materials/finish detail, lighting mood, and exterior scenery visible "
            "through existing windows."
        )
    return (
        f"{task}\n\n"
        "Fail hard if Image 2 adds, removes, moves, resizes, rotates, replaces, "
        "or invents main furniture or architecture; changes the camera/framing; "
        "changes door/window placement; or turns the view into a floorplan/design "
        "diagram. Pretty but structurally wrong images must fail.\n\n"
        f"Case id: {case.id}\n"
        f"Variant id: {variant.id}\n"
        f"Selected style/lighting/scenery presets: {preset_details}\n"
        f"Edit operations: {edit_details}\n\n"
        f"{edit_rules}\n\n"
        f"Visible source objects with bbox and clipping state: {visible_objects}\n"
        f"Openings metadata: {openings}\n\n"
        "Semantic room-shell/crop locks from provider metadata:\n"
        f"{semantic_lock_text}\n\n"
        "If any semantic lock is violated, fail the relevant hard check. Missing, "
        "flattened, completed, or reinterpreted near-crop wall planes must fail "
        "room_shell/camera_framing. Turning wall shadows, seams, or gradients into "
        "new furniture must fail unauthorized_additions and object_inventory. "
        "Changing the count, wall association, apparent size, or silhouette of a "
        "locked opening must fail openings.\n\n"
        "Opening evaluation rules: an item listed under windows confirms that an "
        "existing window is present. A null screenBboxPx means its screen-space "
        "bbox is unavailable, not that the opening is absent. For an existing "
        "window, realistic frame/trim materialization and exterior scenery visible "
        "through the same opening are allowed. Fail openings only for changed "
        "count, placement, apparent size, silhouette, or wall association. Do not "
        "classify allowed window realism or scenery as unauthorized additions.\n\n"
        "Selected style, lighting, and scenery presets are allowed only as finish "
        "controls: material texture, visible style language on existing objects, "
        "lighting mood, and exterior view through existing openings. Do not fail "
        "a candidate only because the coarse blockout became photorealistic, gained "
        "realistic material thickness/detail, or followed the selected preset. "
        "Still fail if preset following causes added/removed/moved main objects, "
        "new openings, changed room shell, changed camera/framing, or changed "
        "visible footprint/scale/depth order.\n\n"
        "Return JSON with keys: status ('passed' or 'failed'), score (0 to 1), "
        "reason, failures (array of short strings), and hard_checks (object). "
        f"hard_checks must contain exactly these keys: {', '.join(required_hard_checks)}. "
        "Each hard check value must be an object with status ('passed' or "
        "'failed'), reason (short string), and violations (array of short strings). "
        "Use failed for any visible mismatch and do not omit uncertain checks."
    )


def _preset_fidelity_eval_prompt(
    *,
    preset_details: Mapping[str, JSONValue],
    source_has_window: bool,
    scenery_applicable: bool,
) -> str:
    details = json.dumps(
        preset_details,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    image_roles = (
        "Image 1 is the 3D layout source and Image 2 is the generated candidate."
        if source_has_window
        else (
            "If two images are provided, Image 1 is the 3D layout source and Image "
            "2 is the generated candidate. If only one image is provided, it is "
            "the generated candidate."
        )
    )
    return (
        f"{image_roles}\n\n"
        "Evaluate only whether the generated candidate follows the selected "
        "render presets. Do not judge layout fidelity, object placement, object "
        "count, camera, or furniture identity here except when a preset is applied "
        "by inventing new architecture or openings.\n\n"
        f"Selected preset details: {details}\n"
        f"Source has a window/opening suitable for scenery: {source_has_window}.\n"
        f"Scenery preset is applicable: {scenery_applicable}.\n\n"
        "Style should be visible through material language, palette, and finish "
        "details on existing visible surfaces/objects. Lighting should match the "
        "requested mood and illumination quality. Scenery should appear only "
        "outside existing windows/openings. If scenery_applicable is false, mark "
        "the scenery check as not_applicable and do not lower the score for missing "
        "scenery; however, fail scenery if the candidate creates a new window, "
        "opening, or exterior view that was not in the source.\n\n"
        "Return JSON with keys: status ('passed' or 'failed'), score (0 to 1), "
        "reason, failures (array of short strings), and checks (object). checks "
        "must contain style, lighting, and scenery. Each check value must be an "
        "object with status ('passed', 'failed', or 'not_applicable'), reason, and "
        "violations (array of short strings). Use failed only for preset-following "
        "problems, not layout problems."
    )


def _extract_chat_json(payload: object) -> dict[str, JSONValue]:
    parsed = _expect_mapping(payload, context="chat_response")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Vision evaluator response has no choices.")
    first_choice = _expect_mapping(choices[0], context="chat_response.choices[0]")
    message = _expect_mapping(first_choice.get("message"), context="message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Vision evaluator response message has no string content.")
    decoded = json.loads(content)
    report = _expect_mapping(decoded, context="vision_report")
    normalized_status = str(report.get("status") or "").strip().lower()
    status = "passed" if normalized_status == "passed" else "failed"
    score = _optional_float(report.get("score"))
    return {
        "status": status,
        "score": 0.0 if math.isnan(score) else score,
        "reason": str(report.get("reason") or ""),
        "failures": _json_safe_list(report.get("failures")),
        "hard_checks": _json_safe(report.get("hard_checks")),
    }


def _extract_preset_chat_json(payload: object) -> dict[str, JSONValue]:
    parsed = _expect_mapping(payload, context="chat_response")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Vision evaluator response has no choices.")
    first_choice = _expect_mapping(choices[0], context="chat_response.choices[0]")
    message = _expect_mapping(first_choice.get("message"), context="message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Vision evaluator response message has no string content.")
    decoded = json.loads(content)
    report = _expect_mapping(decoded, context="preset_vision_report")
    normalized_status = str(report.get("status") or "").strip().lower()
    status = "passed" if normalized_status == "passed" else "failed"
    score = _optional_float(report.get("score"))
    return {
        "status": status,
        "score": 0.0 if math.isnan(score) else score,
        "reason": str(report.get("reason") or ""),
        "failures": _json_safe_list(report.get("failures")),
        "checks": _json_safe(report.get("checks")),
    }


def _hard_check_failures(
    report: Mapping[str, JSONValue],
    required_hard_checks: Sequence[str],
) -> list[JSONValue]:
    raw_checks = report.get("hard_checks")
    checks = raw_checks if isinstance(raw_checks, Mapping) else {}
    failures: list[JSONValue] = []
    for name in required_hard_checks:
        raw_check = checks.get(name)
        check = raw_check if isinstance(raw_check, Mapping) else {}
        status = str(check.get("status") or "").strip().lower()
        if status != "passed":
            failures.append(
                {
                    "check": name,
                    "reason": str(
                        check.get("reason") or "missing_or_failed_hard_check"
                    ),
                    "violations": _json_safe_list(check.get("violations")),
                }
            )
    return failures


def _layout_eval_api_key() -> str | None:
    for name in (_OPENAI_API_KEY_ENV, _OPENAI_IMAGE_API_KEY_ENV):
        value = os.getenv(name, "").strip()
        if value:
            return value
    try:
        from config.openai_config import OpenAIConfig  # noqa: PLC0415

        return OpenAIConfig.OPENAI_API_KEY
    except Exception:
        return None


def _layout_eval_model() -> str:
    configured = os.getenv(_LAYOUT_FIDELITY_MODEL_ENV, "").strip()
    if configured:
        return configured
    try:
        from config.openai_config import OpenAIConfig  # noqa: PLC0415

        return OpenAIConfig.MODELS.helper_model().name
    except Exception:
        return "gpt-4o-mini"


def _encode_image_data_url(path: Path) -> str:
    image_bytes = path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    return (
        f"data:image/{path.suffix.lstrip('.').lower() or 'png'};base64,{image_base64}"
    )


def _torch_module() -> object:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is not installed") from exc
    return torch


def _metric_device_name() -> str:
    requested = os.getenv(_METRIC_DEVICE_ENV, "auto").strip().lower()
    torch = _torch_module()
    if requested and requested != "auto":
        return requested
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _torch_image_tensor(image: Image.Image, *, device: str) -> object:
    torch = _torch_module()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


@lru_cache(maxsize=4)
def _lpips_model() -> tuple[object, str, str]:
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("lpips is not installed") from exc
    net = os.getenv(_LPIPS_NET_ENV, _DEFAULT_LPIPS_NET).strip() or _DEFAULT_LPIPS_NET
    device = _metric_device_name()
    model = lpips.LPIPS(net=net)
    model = model.to(device)
    model.eval()
    return model, device, net


def _dino_model_name() -> str:
    return (
        os.getenv(_DINO_MODEL_ENV, _DEFAULT_DINO_MODEL).strip() or _DEFAULT_DINO_MODEL
    )


@lru_cache(maxsize=2)
def _dino_model_bundle() -> tuple[object, object, str]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError("transformers is not installed") from exc
    model_name = _dino_model_name()
    device = _metric_device_name()
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise OSError(
            f"DINO model is not available locally: {model_name}. "
            f"Set {_DINO_MODEL_ENV} to a local model path."
        ) from exc
    model = model.to(device)
    model.eval()
    return processor, model, device


def _dino_embeddings_for_images(images: Sequence[Image.Image]) -> FloatArray:
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    torch = _torch_module()
    processor, model, device = _dino_model_bundle()
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        normalized = torch.nn.functional.normalize(pooled, dim=-1)
    return cast(FloatArray, normalized.detach().cpu().numpy().astype(np.float32))


def _dino_patch_embeddings_for_images(images: Sequence[Image.Image]) -> FloatArray:
    if not images:
        return np.empty((0, 0, 0), dtype=np.float32)
    torch = _torch_module()
    processor, model, device = _dino_model_bundle()
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        hidden_state = outputs.last_hidden_state
        if hidden_state.shape[1] <= 1:
            raise RuntimeError("DINO output has no patch tokens.")
        patch_tokens = hidden_state[:, 1:, :]
        normalized = torch.nn.functional.normalize(patch_tokens, dim=-1)
    return cast(FloatArray, normalized.detach().cpu().numpy().astype(np.float32))


def _depth_model_name() -> str:
    return (
        os.getenv(_DEPTH_MODEL_ENV, _DEFAULT_DEPTH_MODEL).strip()
        or _DEFAULT_DEPTH_MODEL
    )


@lru_cache(maxsize=2)
def _depth_model_bundle() -> tuple[object, object, str]:
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise ImportError("transformers is not installed") from exc
    model_name = _depth_model_name()
    device = _metric_device_name()
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            local_files_only=True,
        )
    except (OSError, ValueError) as exc:
        raise OSError(
            f"Depth model is not available locally: {model_name}. "
            f"Set {_DEPTH_MODEL_ENV} to a local model path."
        ) from exc
    model = model.to(device)
    model.eval()
    return processor, model, device


def _depth_maps_for_images(images: Sequence[Image.Image]) -> FloatArray:
    if not images:
        return np.empty((0, 0, 0), dtype=np.float32)
    torch = _torch_module()
    processor, model, device = _depth_model_bundle()
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        predicted_depth = getattr(outputs, "predicted_depth", None)
        if predicted_depth is None:
            raise RuntimeError("Depth model output has no predicted_depth.")
        if predicted_depth.ndim == 4:
            predicted_depth = predicted_depth.squeeze(1)
    return cast(FloatArray, predicted_depth.detach().cpu().numpy().astype(np.float32))


def _yolo_model_name() -> str:
    return (
        os.getenv(_YOLO_MODEL_ENV, _DEFAULT_YOLO_MODEL).strip() or _DEFAULT_YOLO_MODEL
    )


@lru_cache(maxsize=2)
def _yolo_model() -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed") from exc
    model_name = _yolo_model_name()
    try:
        return YOLO(model_name)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OSError(
            f"YOLO model is not available: {model_name}. "
            f"Set {_YOLO_MODEL_ENV} to a local model path."
        ) from exc


def _detect_objects(image_path: Path) -> list[DetectionBox]:
    model = _yolo_model()
    results = model.predict(
        source=str(image_path),
        verbose=False,
        conf=_DETECTOR_MIN_CONFIDENCE,
        device=_metric_device_name(),
    )
    detections: list[DetectionBox] = []
    if not results:
        return detections
    first = results[0]
    boxes = getattr(first, "boxes", None)
    names = getattr(first, "names", {})
    if boxes is None:
        return detections
    xyxyn = getattr(boxes, "xyxyn", None)
    cls_values = getattr(boxes, "cls", None)
    conf_values = getattr(boxes, "conf", None)
    if xyxyn is None or cls_values is None or conf_values is None:
        return detections
    for bbox_tensor, cls_tensor, conf_tensor in zip(xyxyn, cls_values, conf_values):
        bbox = [float(value) for value in bbox_tensor.detach().cpu().tolist()]
        label_id = int(cls_tensor.detach().cpu().item())
        label = str(names.get(label_id, label_id)).lower().strip()
        detections.append(
            DetectionBox(
                label=label,
                confidence=float(conf_tensor.detach().cpu().item()),
                bbox_norm={
                    "x": _clamp01(bbox[0]),
                    "y": _clamp01(bbox[1]),
                    "width": _clamp01(bbox[2] - bbox[0]),
                    "height": _clamp01(bbox[3] - bbox[1]),
                },
            )
        )
    return detections


def _edit_bbox_mask(
    size: tuple[int, int],
    operations: Sequence[EditOperationConfig],
    *,
    margin_ratio: float = 0.0,
) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    for operation in operations:
        bbox = operation.bbox_norm
        if bbox is None:
            continue
        left = int(max(0.0, min(1.0, bbox.x - margin_ratio)) * width)
        top = int(max(0.0, min(1.0, bbox.y - margin_ratio)) * height)
        right = int(max(0.0, min(1.0, bbox.x + bbox.width + margin_ratio)) * width)
        bottom = int(max(0.0, min(1.0, bbox.y + bbox.height + margin_ratio)) * height)
        if right > left and bottom > top:
            mask.paste(255, box=(left, top, right, bottom))
    return mask


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _expect_mapping(payload, context=str(path))


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object for {context}.")
    return cast(Mapping[str, object], value)


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _json_safe_list(value: object) -> list[JSONValue]:
    if not isinstance(value, list):
        return []
    return [_json_safe(item) for item in value]


def _json_value_mapping(value: object) -> Mapping[str, JSONValue] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): _json_safe(item) for key, item in value.items()}


def _match_expected_detections(
    expected: Sequence[VisibleObjectSpec],
    detections: Sequence[DetectionBox],
) -> dict[str, JSONValue]:
    matched: dict[str, JSONValue] = {}
    unmatched_detections = list(detections)
    missing: list[JSONValue] = []
    center_distances: list[float] = []
    scale_deltas: list[float] = []
    ious: list[float] = []
    for item in expected:
        candidates = [
            detection
            for detection in unmatched_detections
            if detection.label == item.detector_label
        ]
        if not candidates:
            missing.append(
                {
                    "object_id": item.object_id,
                    "object_type": item.object_type,
                    "detector_label": item.detector_label,
                    "reason": "label_not_detected",
                }
            )
            continue
        best = min(
            candidates,
            key=lambda detection: _center_distance(item.bbox_norm, detection.bbox_norm),
        )
        unmatched_detections.remove(best)
        center_distance = _center_distance(item.bbox_norm, best.bbox_norm)
        scale_delta = _scale_delta(item.bbox_norm, best.bbox_norm)
        iou = _bbox_iou(item.bbox_norm, best.bbox_norm)
        center_distances.append(center_distance)
        scale_deltas.append(scale_delta)
        ious.append(iou)
        matched[item.object_id] = {
            "object_id": item.object_id,
            "object_type": item.object_type,
            "detector_label": item.detector_label,
            "expected_bbox_norm": dict(item.bbox_norm),
            "detected_bbox_norm": dict(best.bbox_norm),
            "confidence": round(best.confidence, 6),
            "center_distance": round(center_distance, 6),
            "scale_delta": round(scale_delta, 6),
            "iou": round(iou, 6),
        }

    expected_count = len(expected)
    matched_count = len(matched)
    return {
        "expected_count": expected_count,
        "matched_count": matched_count,
        "missing_count": expected_count - matched_count,
        "match_ratio": matched_count / max(1, expected_count),
        "mean_center_distance": _mean(center_distances),
        "mean_scale_delta": _mean(scale_deltas),
        "mean_iou": _mean(ious),
        "matched_objects": matched,
        "missing_objects": missing,
        "detected_count": len(detections),
    }


def _spatial_relation_stats(
    expected: Sequence[VisibleObjectSpec],
    matched: Mapping[str, object],
) -> dict[str, JSONValue]:
    relations: list[dict[str, JSONValue]] = []
    for index, left in enumerate(expected):
        left_match = _expect_optional_mapping(matched.get(left.object_id))
        if left_match is None:
            continue
        for right in expected[index + 1 :]:
            right_match = _expect_optional_mapping(matched.get(right.object_id))
            if right_match is None:
                continue
            expected_relation = _dominant_spatial_relation(
                left.bbox_norm, right.bbox_norm
            )
            if expected_relation is None:
                continue
            left_detected = _expect_optional_mapping(
                left_match.get("detected_bbox_norm")
            )
            right_detected = _expect_optional_mapping(
                right_match.get("detected_bbox_norm")
            )
            if left_detected is None or right_detected is None:
                continue
            detected_relation = _dominant_spatial_relation(
                left_detected, right_detected
            )
            matched_relation = expected_relation == detected_relation
            relations.append(
                {
                    "object_a": left.object_id,
                    "object_b": right.object_id,
                    "expected_relation": expected_relation,
                    "detected_relation": detected_relation,
                    "matched": matched_relation,
                }
            )
    if not relations:
        return {
            "relation_count": 0,
            "matched_relation_count": 0,
            "relation_match_ratio": 0.0,
            "relations": [],
            "reason": "no_matched_unambiguous_relations",
        }
    matched_count = sum(1 for item in relations if item["matched"])
    return {
        "relation_count": len(relations),
        "matched_relation_count": matched_count,
        "relation_match_ratio": matched_count / len(relations),
        "relations": relations,
    }


def _detector_label_for_object(label: str) -> str | None:
    normalized = _normalize_label(label)
    aliases = {
        "armchair": "chair",
        "bookshelf": "book",
        "chair": "chair",
        "coffee_table": "dining table",
        "desk": "dining table",
        "desk_chair": "chair",
        "dining_table": "dining table",
        "floor_lamp": "lamp",
        "lamp": "lamp",
        "lounge_chair": "chair",
        "monitor": "tv",
        "ottoman": "chair",
        "plant": "potted plant",
        "side_chair": "chair",
        "side_table": "dining table",
        "sofa": "couch",
        "stool": "chair",
        "table": "dining table",
        "tv": "tv",
        "tv_console": "tv",
        "television": "tv",
    }
    return aliases.get(normalized)


def _normalize_label(label: str) -> str:
    cleaned = label.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _expect_optional_mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None


def _dominant_spatial_relation(
    left_bbox: Mapping[str, object],
    right_bbox: Mapping[str, object],
) -> str | None:
    left_x, left_y = _bbox_center(left_bbox)
    right_x, right_y = _bbox_center(right_bbox)
    delta_x = right_x - left_x
    delta_y = right_y - left_y
    if (
        abs(delta_x) < _SPATIAL_RELATION_MIN_DELTA
        and abs(delta_y) < _SPATIAL_RELATION_MIN_DELTA
    ):
        return None
    if abs(delta_x) >= abs(delta_y):
        return "left_of" if delta_x > 0 else "right_of"
    return "above" if delta_y > 0 else "below"


def _bbox_center(bbox: Mapping[str, object]) -> tuple[float, float]:
    x = _optional_float(bbox.get("x"))
    y = _optional_float(bbox.get("y"))
    width = _optional_float(bbox.get("width"))
    height = _optional_float(bbox.get("height"))
    return x + width / 2.0, y + height / 2.0


def _center_distance(
    expected: Mapping[str, object],
    detected: Mapping[str, object],
) -> float:
    expected_x, expected_y = _bbox_center(expected)
    detected_x, detected_y = _bbox_center(detected)
    return math.sqrt((expected_x - detected_x) ** 2 + (expected_y - detected_y) ** 2)


def _scale_delta(
    expected: Mapping[str, object],
    detected: Mapping[str, object],
) -> float:
    expected_area = _bbox_area(expected)
    detected_area = _bbox_area(detected)
    if expected_area <= 0.0:
        return 1.0
    return abs(detected_area - expected_area) / expected_area


def _bbox_area(bbox: Mapping[str, object]) -> float:
    return max(0.0, _optional_float(bbox.get("width"))) * max(
        0.0,
        _optional_float(bbox.get("height")),
    )


def _bbox_iou(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> float:
    left_x = _optional_float(left.get("x"))
    left_y = _optional_float(left.get("y"))
    left_right = left_x + _optional_float(left.get("width"))
    left_bottom = left_y + _optional_float(left.get("height"))
    right_x = _optional_float(right.get("x"))
    right_y = _optional_float(right.get("y"))
    right_right = right_x + _optional_float(right.get("width"))
    right_bottom = right_y + _optional_float(right.get("height"))
    intersection_width = max(0.0, min(left_right, right_right) - max(left_x, right_x))
    intersection_height = max(
        0.0, min(left_bottom, right_bottom) - max(left_y, right_y)
    )
    intersection = intersection_width * intersection_height
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _padded_crop_box(
    bbox: Mapping[str, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    bbox_width = max(0.0, bbox["maxX"] - bbox["minX"])
    bbox_height = max(0.0, bbox["maxY"] - bbox["minY"])
    padding = max(2.0, max(bbox_width, bbox_height) * _OBJECT_METRIC_CROP_PADDING_RATIO)
    left = max(0, math.floor(bbox["minX"] - padding))
    top = max(0, math.floor(bbox["minY"] - padding))
    right = min(width, math.ceil(bbox["maxX"] + padding))
    bottom = min(height, math.ceil(bbox["maxY"] + padding))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _cosine_similarity(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_number(value: object) -> float | None:
    parsed = _optional_float(value)
    return None if math.isnan(parsed) else parsed


def _open_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _rgb_array(image: Image.Image) -> UInt8Array:
    return cast(UInt8Array, np.asarray(image.convert("RGB"), dtype=np.uint8))


def _load_selected_mask(
    *,
    mask_path: Path | None,
    size: tuple[int, int],
    selected_region: str,
) -> BoolMask | None:
    if mask_path is None or selected_region == "all":
        return None
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    if selected_region == "inside_mask":
        return cast(BoolMask, mask_array)
    if selected_region == "outside_mask":
        return cast(BoolMask, np.logical_not(mask_array))
    raise ValueError(f"Unknown selected region: {selected_region}")


def _selected_mask_or_all(mask: BoolMask | None, shape: tuple[int, int]) -> BoolMask:
    if mask is not None:
        return mask
    return cast(BoolMask, np.ones(shape, dtype=np.bool_))


def _selected_pixels(
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
) -> tuple[FloatArray, FloatArray]:
    selected = _selected_mask_or_all(selected_mask, reference_rgb.shape[:2])
    return (
        cast(FloatArray, reference_rgb[selected].astype(np.float32)),
        cast(FloatArray, candidate_rgb[selected].astype(np.float32)),
    )


def _write_diff_image(
    *,
    diff: NDArray[np.int16],
    selected_mask: BoolMask,
    output_dir: Path,
    case_id: str,
    variant_id: str,
    prefix: str,
) -> Path:
    visible_diff = diff.copy()
    visible_diff[np.logical_not(selected_mask)] = 0
    enhanced = np.clip(visible_diff * 4, 0, 255).astype(np.uint8)
    path = output_dir / case_id / variant_id / f"{prefix}_diff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced, mode="RGB").save(path)
    return path


def _corner_background_rgb(rgb: FloatArray) -> FloatArray:
    height, width = rgb.shape[:2]
    patch = max(1, min(height, width, 8))
    samples = np.concatenate(
        [
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, width - patch :].reshape(-1, 3),
            rgb[height - patch :, :patch].reshape(-1, 3),
            rgb[height - patch :, width - patch :].reshape(-1, 3),
        ],
        axis=0,
    )
    return cast(FloatArray, np.median(samples, axis=0))


def _threshold(
    thresholds: Mapping[str, float],
    prefix: str,
    key: str,
) -> float | None:
    for candidate in (f"{prefix}.{key}", f"{prefix}_{key}", key):
        if candidate in thresholds:
            return thresholds[candidate]
    return None


def _status_for_threshold(
    *,
    value: float,
    threshold: float | None,
    direction: MetricDirection,
) -> MetricStatus:
    if threshold is None or direction == "none":
        return "warning"
    if direction == "min":
        return "passed" if value >= threshold else "failed"
    return "passed" if value <= threshold else "failed"


def _skipped(name: str, reason: str) -> MetricResult:
    return MetricResult(name=name, status="skipped", details={"reason": reason})


def _optional_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean(values: Iterable[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _rate(values: Iterable[bool]) -> float | None:
    clean = list(values)
    if not clean:
        return None
    return sum(1 for value in clean if value) / len(clean)
