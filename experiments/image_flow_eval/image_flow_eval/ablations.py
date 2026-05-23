from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .types import CaseConfig, EditOperationConfig, VariantConfig


@dataclass(frozen=True)
class AblatedInputs:
    render_prompt: str | None
    edit_prompt: str | None
    layout_reference_image: Path | None
    annotated_reference_image: Path | None
    scene_reference_image: Path | None
    edit_operations: tuple[EditOperationConfig, ...]


def apply_ablation(case: CaseConfig, variant: VariantConfig) -> AblatedInputs:
    edit_prompt = variant.prompt or case.edit_prompt
    render_prompt = variant.prompt or case.render_prompt
    layout_reference_image = case.layout_reference_image
    annotated_reference_image = case.annotated_reference_image
    scene_reference_image = case.scene_reference_image
    edit_operations = case.edit_operations

    if variant.mode == "no_layout_reference":
        layout_reference_image = None
        annotated_reference_image = None

    if variant.mode == "no_edit_mask":
        edit_operations = tuple(
            replace(operation, bbox_norm=None) for operation in edit_operations
        )

    if variant.mode == "no_preservation_prompt":
        requested = edit_prompt or render_prompt or "Apply the requested image change."
        edit_prompt = f"Apply this image edit: {requested}"
        render_prompt = requested

    if variant.mode == "no_reference_images":
        scene_reference_image = None
        edit_operations = tuple(
            replace(operation, replacement_image=None) for operation in edit_operations
        )

    return AblatedInputs(
        render_prompt=render_prompt,
        edit_prompt=edit_prompt,
        layout_reference_image=layout_reference_image,
        annotated_reference_image=annotated_reference_image,
        scene_reference_image=scene_reference_image,
        edit_operations=edit_operations,
    )
