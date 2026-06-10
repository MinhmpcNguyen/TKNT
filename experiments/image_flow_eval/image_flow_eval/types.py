from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

VariantMode = Literal[
    "baseline",
    "simple_prompt",
    "simple_prompt_with_instance_overlay",
    "contract_lite",
    "contract_objects",
    "contract_objects_preflight",
    "contract_objects_preflight_locked_geometry",
    "contract_objects_preflight_visual_only",
    "llm_visual_preflight",
    "llm_visual_preflight_with_instance_overlay",
    "llm_visual_preflight_object_anchors",
    "llm_visual_preflight_no_metadata",
    "single_edit_edge_silhouette_guided",
    "single_edit_geometry_locked",
    "two_pass_geometry_repair",
    "edit_prompt_image_only",
    "edit_object_scope_preflight",
    "operation_dsl",
    "operation_dsl_preflight",
    "operation_dsl_preflight_no_dsl",
    "operation_dsl_preflight_no_edit_mask",
    "no_layout_reference",
    "no_edit_mask",
    "no_preservation_prompt",
    "no_reference_images",
    "no_preset_reference_images",
]
ProviderName = Literal["evaluate_only", "tknt_snapshot"]
ProviderStatus = Literal[
    "generated",
    "reused_approved_render",
    "reused_existing",
    "reused_render_partial",
]
VisualPreflightMode = Literal["disabled", "snapshot_and_metadata", "snapshot_only"]
VisualPreflightProfile = Literal[
    "classic",
    "room_shell",
    "object_anchors",
    "edit_scope",
    "edit_object_scope",
]
EditOperationType = Literal["prompt", "add", "replace", "recolor", "remove"]
EditSourceSelectionMode = Literal["best_passed", "exact"]
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
    operation_type: EditOperationType | None = None
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
class RenderPresetSelection:
    style: str | None = None
    lighting: str | None = None
    scenery: str | None = None

    def to_mapping(self) -> Mapping[str, str]:
        selected: dict[str, str] = {}
        if self.style:
            selected["style"] = self.style
        if self.lighting:
            selected["lighting"] = self.lighting
        if self.scenery:
            selected["scenery"] = self.scenery
        return selected


@dataclass(frozen=True)
class EditSourceVariantConfig:
    case_id: str
    variant: str
    selection_mode: EditSourceSelectionMode = "best_passed"


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
    preset_selection: RenderPresetSelection | None = None
    edit_source_variant: EditSourceVariantConfig | None = None
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
    source_image: Path | None = None
    source_snapshot_payload: Path | None = None
    approved_render_source: Path | None = None
    provider_status: ProviderStatus = "generated"
    runtime_s: float | None = None


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
