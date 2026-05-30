from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

VariantMode = Literal[
    "baseline",
    "no_layout_reference",
    "no_edit_mask",
    "no_preservation_prompt",
    "no_reference_images",
]
ProviderName = Literal["evaluate_only", "tknt_snapshot"]
AlignMode = Literal["strict", "resize_to_reference"]
MetricStatus = Literal["passed", "failed", "warning", "skipped"]
MetricDirection = Literal["min", "max", "none"]


@dataclass(frozen=True)
class BBoxNorm:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class EditOperationConfig:
    object_id: str
    object_name: str | None = None
    operation_type: Literal["prompt", "add", "replace", "recolor"] | None = None
    prompt: str | None = None
    bbox_norm: BBoxNorm | None = None
    replacement_image: Path | None = None
    target_color: str | None = None


@dataclass(frozen=True)
class VariantConfig:
    id: str
    mode: VariantMode = "baseline"
    provider: ProviderName | None = None
    output_image: Path | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class CaseConfig:
    id: str
    source_image: Path | None = None
    reference_image: Path | None = None
    target_image: Path | None = None
    mask_image: Path | None = None
    render_prompt: str | None = None
    edit_prompt: str | None = None
    snapshot_payload: Path | None = None
    snapshot_image: Path | None = None
    layout_reference_image: Path | None = None
    annotated_reference_image: Path | None = None
    scene_reference_image: Path | None = None
    edit_operations: tuple[EditOperationConfig, ...] = ()
    variants: tuple[VariantConfig, ...] = ()


@dataclass(frozen=True)
class ManifestConfig:
    path: Path
    output_dir: Path | None
    metrics: tuple[str, ...]
    thresholds: Mapping[str, float]
    visual_delta_threshold: int
    align_mode: AlignMode
    cases: tuple[CaseConfig, ...]


@dataclass(frozen=True)
class ProviderResult:
    case_id: str
    variant_id: str
    output_image: Path
    metadata_path: Path | None
    render_image: Path | None = None


@dataclass(frozen=True)
class MetricResult:
    name: str
    status: MetricStatus
    value: float | None = None
    threshold: float | None = None
    direction: MetricDirection = "none"
    details: Mapping[str, JSONValue] | None = None

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "details": dict(self.details or {}),
        }
