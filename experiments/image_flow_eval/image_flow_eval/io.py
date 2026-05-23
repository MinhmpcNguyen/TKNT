from __future__ import annotations

import base64
import csv
import json
import mimetypes
import shlex
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from .types import (
    AlignMode,
    BBoxNorm,
    CaseConfig,
    EditOperationConfig,
    JSONValue,
    ManifestConfig,
    MetricResult,
    ProviderName,
    VariantConfig,
    VariantMode,
)

_DEFAULT_METRICS: tuple[str, ...] = (
    "blank",
    "visual_diff",
    "psnr",
    "ssim",
    "nrmse",
    "lpips",
    "clip",
)
_VALID_VARIANT_MODES: set[str] = {
    "baseline",
    "no_layout_reference",
    "no_edit_mask",
    "no_preservation_prompt",
    "no_reference_images",
}
_VALID_PROVIDERS: set[str] = {"evaluate_only", "tknt_snapshot", "command"}
_VALID_ALIGN_MODES: set[str] = {"strict", "resize_to_reference"}


def load_manifest(path: str | Path) -> ManifestConfig:
    manifest_path = Path(path).expanduser().resolve()
    payload = _expect_mapping(read_json_file(manifest_path), context=str(manifest_path))
    base_dir = manifest_path.parent
    defaults = _expect_mapping(payload.get("defaults", {}), context="defaults")

    metrics = tuple(_string_sequence(defaults.get("metrics"), fallback=_DEFAULT_METRICS))
    thresholds = _float_mapping(defaults.get("thresholds", {}))
    visual_delta_threshold = _int_value(
        defaults.get("visual_delta_threshold"),
        fallback=8,
    )
    align_mode = _align_mode(_string_value(defaults.get("align_mode"), "resize_to_reference"))
    output_dir = _optional_path(base_dir, payload.get("output_dir"))
    fid_reference_images = tuple(
        _path(base_dir, item) for item in _string_sequence(payload.get("fid_reference_images"))
    )

    cases = tuple(
        _parse_case(base_dir, item)
        for item in _mapping_sequence(payload.get("cases"), context="cases")
    )
    return ManifestConfig(
        path=manifest_path,
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        visual_delta_threshold=visual_delta_threshold,
        align_mode=align_mode,
        fid_reference_images=fid_reference_images,
        cases=cases,
    )


def read_json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Mapping[str, JSONValue] | Sequence[JSONValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_metric_csv(path: Path, rows: Iterable[Mapping[str, JSONValue]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "variant_id",
        "metric",
        "status",
        "value",
        "threshold",
        "direction",
        "details",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            details = row.get("details")
            serialized = dict(row)
            serialized["details"] = (
                json.dumps(details, ensure_ascii=False, sort_keys=True)
                if isinstance(details, dict)
                else ""
            )
            writer.writerow(serialized)


def encode_image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    image_base64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"


def write_data_url_image(data_url: str, output_path: Path) -> Path:
    if not data_url.startswith("data:image/") or "," not in data_url:
        raise ValueError("Expected an image data URL.")
    header, image_base64 = data_url.split(",", maxsplit=1)
    mime_type = header.removeprefix("data:").split(";", maxsplit=1)[0]
    suffix = _suffix_for_mime_type(mime_type)
    resolved_path = output_path.with_suffix(suffix)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_bytes(base64.b64decode(image_base64))
    return resolved_path


def metric_results_to_json(results: Sequence[MetricResult]) -> list[JSONValue]:
    return [result.to_json() for result in results]


def _parse_case(base_dir: Path, value: Mapping[str, object]) -> CaseConfig:
    case_id = _required_string(value, "id", context="case")
    variants = tuple(
        _parse_variant(base_dir, item)
        for item in _mapping_sequence(value.get("variants", []), context=f"{case_id}.variants")
    )
    if not variants:
        variants = (
            VariantConfig(
                id="baseline",
                mode="baseline",
                provider="evaluate_only",
                output_image=_optional_path(base_dir, value.get("output_image")),
            ),
        )
    return CaseConfig(
        id=case_id,
        source_image=_optional_path(base_dir, value.get("source_image")),
        reference_image=_optional_path(base_dir, value.get("reference_image")),
        target_image=_optional_path(base_dir, value.get("target_image")),
        mask_image=_optional_path(base_dir, value.get("mask_image")),
        target_caption=_optional_string(value.get("target_caption")),
        render_prompt=_optional_string(value.get("render_prompt")),
        edit_prompt=_optional_string(value.get("edit_prompt")),
        snapshot_payload=_optional_path(base_dir, value.get("snapshot_payload")),
        snapshot_image=_optional_path(base_dir, value.get("snapshot_image")),
        layout_reference_image=_optional_path(base_dir, value.get("layout_reference_image")),
        annotated_reference_image=_optional_path(base_dir, value.get("annotated_reference_image")),
        scene_reference_image=_optional_path(base_dir, value.get("scene_reference_image")),
        edit_operations=tuple(
            _parse_edit_operation(base_dir, item)
            for item in _mapping_sequence(
                value.get("edit_operations", []),
                context=f"{case_id}.edit_operations",
            )
        ),
        variants=variants,
    )


def _parse_variant(base_dir: Path, value: Mapping[str, object]) -> VariantConfig:
    variant_id = _required_string(value, "id", context="variant")
    mode_raw = _string_value(value.get("mode"), variant_id)
    if mode_raw not in _VALID_VARIANT_MODES:
        mode_raw = "baseline"
    provider_raw = _optional_string(value.get("provider"))
    provider = _provider_name(provider_raw) if provider_raw else None
    command = _optional_command(value.get("command"))
    return VariantConfig(
        id=variant_id,
        mode=cast(VariantMode, mode_raw),
        provider=provider,
        output_image=_optional_path(base_dir, value.get("output_image")),
        command=command,
        prompt=_optional_string(value.get("prompt")),
    )


def _parse_edit_operation(base_dir: Path, value: Mapping[str, object]) -> EditOperationConfig:
    bbox = value.get("bbox_norm")
    return EditOperationConfig(
        object_id=_required_string(value, "object_id", context="edit_operation"),
        object_name=_optional_string(value.get("object_name")),
        operation_type=_optional_operation_type(value.get("operation_type")),
        prompt=_optional_string(value.get("prompt")),
        bbox_norm=_parse_bbox_norm(bbox) if bbox is not None else None,
        replacement_image=_optional_path(base_dir, value.get("replacement_image")),
        target_color=_optional_string(value.get("target_color")),
    )


def _parse_bbox_norm(value: object) -> BBoxNorm:
    payload = _expect_mapping(value, context="bbox_norm")
    return BBoxNorm(
        x=_float_value(payload.get("x"), field="bbox_norm.x"),
        y=_float_value(payload.get("y"), field="bbox_norm.y"),
        width=_float_value(payload.get("width"), field="bbox_norm.width"),
        height=_float_value(payload.get("height"), field="bbox_norm.height"),
    )


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object for {context}.")
    return cast(Mapping[str, object], value)


def _mapping_sequence(value: object, *, context: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {context}.")
    return [_expect_mapping(item, context=f"{context}[]") for item in value]


def _string_sequence(value: object, fallback: Sequence[str] = ()) -> list[str]:
    if value is None:
        return list(fallback)
    if not isinstance(value, list):
        raise ValueError("Expected list of strings.")
    return [str(item) for item in value]


def _float_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    payload = _expect_mapping(value, context="thresholds")
    return {key: _float_value(item, field=key) for key, item in payload.items()}


def _optional_command(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise ValueError("Variant command must be a string or list of strings.")


def _optional_operation_type(
    value: object,
) -> Literal["prompt", "add", "replace", "recolor"] | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized not in {"prompt", "add", "replace", "recolor"}:
        raise ValueError(f"Unknown edit operation type: {normalized}")
    return cast(Literal["prompt", "add", "replace", "recolor"], normalized)


def _provider_name(value: str) -> ProviderName:
    if value not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown provider {value!r}.")
    return cast(ProviderName, value)


def _align_mode(value: str) -> AlignMode:
    if value not in _VALID_ALIGN_MODES:
        raise ValueError(f"Unknown align mode {value!r}.")
    return cast(AlignMode, value)


def _optional_path(base_dir: Path, value: object) -> Path | None:
    text = _optional_string(value)
    if text is None:
        return None
    return _path(base_dir, text)


def _path(base_dir: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _required_string(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string {context}.{key}.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_value(value: object, fallback: str) -> str:
    text = _optional_string(value)
    return text if text is not None else fallback


def _float_value(value: object, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value for {field}.") from exc


def _int_value(value: object, *, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected integer value.") from exc


def _suffix_for_mime_type(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "image/gif":
        return ".gif"
    return ".png"
