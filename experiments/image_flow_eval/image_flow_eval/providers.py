from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .ablations import apply_ablation
from .io import encode_image_to_data_url, write_data_url_image
from .types import (
    BBoxNorm,
    CaseConfig,
    JSONValue,
    ProviderName,
    ProviderResult,
    VariantConfig,
)


def run_case_variant(
    *,
    provider: ProviderName,
    case: CaseConfig,
    variant: VariantConfig,
    output_root: Path,
) -> ProviderResult:
    if provider == "evaluate_only":
        return _evaluate_only(case=case, variant=variant)
    return _run_tknt_snapshot(case=case, variant=variant, output_root=output_root)


def resolve_provider(
    default_provider: ProviderName, variant: VariantConfig
) -> ProviderName:
    if variant.provider is not None:
        return variant.provider
    if variant.output_image is not None:
        return "evaluate_only"
    return default_provider


def _evaluate_only(*, case: CaseConfig, variant: VariantConfig) -> ProviderResult:
    if variant.output_image is None:
        raise ValueError(f"Variant {variant.id} in case {case.id} has no output_image.")
    if not variant.output_image.exists():
        raise FileNotFoundError(f"Output image not found: {variant.output_image}")
    return ProviderResult(
        case_id=case.id,
        variant_id=variant.id,
        output_image=variant.output_image,
        metadata_path=None,
    )


def _run_tknt_snapshot(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_root: Path,
) -> ProviderResult:
    if case.snapshot_payload is None:
        raise ValueError(f"Case {case.id} requires snapshot_payload for tknt_snapshot.")
    if case.snapshot_image is None:
        raise ValueError(f"Case {case.id} requires snapshot_image for tknt_snapshot.")
    if not case.snapshot_payload.exists():
        raise FileNotFoundError(f"Snapshot payload not found: {case.snapshot_payload}")
    if not case.snapshot_image.exists():
        raise FileNotFoundError(f"Snapshot image not found: {case.snapshot_image}")

    from pipeline.snapshot_image_renderer import (
        SnapshotEditOperation,
        render_snapshot_image_from_path,
    )

    ablated = apply_ablation(case, variant)
    variant_dir = output_root / case.id / variant.id
    variant_dir.mkdir(parents=True, exist_ok=True)

    snapshot_image_data_url = encode_image_to_data_url(case.snapshot_image)
    layout_reference_data_url = _optional_data_url(ablated.layout_reference_image)
    annotated_reference_data_url = _optional_data_url(ablated.annotated_reference_image)
    scene_reference_data_url = _optional_data_url(ablated.scene_reference_image)

    render_result = render_snapshot_image_from_path(
        case.snapshot_payload,
        snapshot_image_data_url=snapshot_image_data_url,
        user_prompt=ablated.render_prompt,
        layout_reference_image_data_url=layout_reference_data_url,
        annotated_reference_image_data_url=annotated_reference_data_url,
        scene_reference_image_data_url=scene_reference_data_url,
        scene_reference_mode="target_layout_with_scene_reference"
        if scene_reference_data_url
        else "none",
        render_mode="generate",
        trace_id=f"{case.id}:{variant.id}:generate",
    )
    render_data_url = _extract_result_data_url(render_result)
    render_image = write_data_url_image(render_data_url, variant_dir / "render.png")

    edit_operations = [
        SnapshotEditOperation(
            object_id=operation.object_id,
            object_name=operation.object_name,
            operation_type=operation.operation_type,
            prompt=operation.prompt,
            bbox_norm=_bbox_mapping(operation.bbox_norm),
            replacement_image_data_url=_optional_data_url(operation.replacement_image),
            target_color=operation.target_color,
        )
        for operation in ablated.edit_operations
    ]
    should_edit = bool(ablated.edit_prompt or edit_operations)
    output_image = render_image
    edit_result: Mapping[str, object] | None = None
    if should_edit:
        edit_result = render_snapshot_image_from_path(
            case.snapshot_payload,
            snapshot_image_data_url=snapshot_image_data_url,
            user_prompt=ablated.edit_prompt,
            layout_reference_image_data_url=layout_reference_data_url,
            annotated_reference_image_data_url=annotated_reference_data_url,
            edit_source_image_data_url=render_data_url,
            render_mode="edit",
            edit_operations=edit_operations,
            trace_id=f"{case.id}:{variant.id}:edit",
        )
        output_image = write_data_url_image(
            _extract_result_data_url(edit_result),
            variant_dir / "edit.png",
        )

    metadata_path = variant_dir / "tknt_snapshot_result.json"
    metadata_path.write_text(
        json.dumps(
            {
                "case_id": case.id,
                "variant_id": variant.id,
                "mode": variant.mode,
                "render_image": str(render_image),
                "output_image": str(output_image),
                "render_result": _summarize_render_result(render_result),
                "edit_result": _summarize_render_result(edit_result)
                if edit_result is not None
                else None,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return ProviderResult(
        case_id=case.id,
        variant_id=variant.id,
        output_image=output_image,
        metadata_path=metadata_path,
        render_image=render_image,
    )


def _optional_data_url(path: Path | None) -> str | None:
    if path is None:
        return None
    return encode_image_to_data_url(path)


def _bbox_mapping(bbox: BBoxNorm | None) -> Mapping[str, float] | None:
    if bbox is None:
        return None
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }


def _extract_result_data_url(result: Mapping[str, object]) -> str:
    image_payload = result.get("image")
    if not isinstance(image_payload, dict):
        raise ValueError("Render result is missing image payload.")
    data_url = image_payload.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ValueError("Render result image payload has no data_url.")
    return data_url


def _summarize_render_result(result: Mapping[str, object] | None) -> JSONValue:
    if result is None:
        return None
    image_payload = _mapping(result.get("image"))
    return {
        "request": _json_safe(result.get("request")),
        "models": _json_safe(result.get("models")),
        "metadata": _json_safe(result.get("metadata")),
        "image": {
            "mime_type": _json_safe(image_payload.get("mime_type")),
            "size": _json_safe(image_payload.get("size")),
            "quality": _json_safe(image_payload.get("quality")),
            "aspect_ratio": _json_safe(image_payload.get("aspect_ratio")),
            "retry_count": _json_safe(image_payload.get("retry_count")),
        },
    }


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
