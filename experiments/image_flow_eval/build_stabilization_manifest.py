from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from experiments.image_flow_eval.image_flow_eval.types import JSONValue

logger = logging.getLogger(__name__)

Profile = Literal["compact_reading", "guest_vanity"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CASES_ROOT = _REPO_ROOT / "experiments" / "image_flow_eval" / "cases"
_RUNS_ROOT = _REPO_ROOT / "experiments" / "image_flow_eval" / "runs"
_CREATE_VARIANTS: list[JSONValue] = [
    {
        "id": "contract_objects_preflight_locked_geometry",
        "mode": "contract_objects_preflight_locked_geometry",
    },
    {
        "id": "llm_visual_preflight",
        "mode": "llm_visual_preflight",
    },
    {
        "id": "llm_visual_preflight_object_anchors",
        "mode": "llm_visual_preflight_object_anchors",
    },
    {
        "id": "single_edit_edge_silhouette_guided",
        "mode": "single_edit_edge_silhouette_guided",
    },
    {
        "id": "contract_objects_preflight",
        "mode": "contract_objects_preflight",
    },
    {"id": "contract_objects", "mode": "contract_objects"},
    {"id": "simple_prompt", "mode": "simple_prompt"},
]
_FULL_EDIT_VARIANT: list[JSONValue] = [
    {"id": "operation_dsl_preflight", "mode": "operation_dsl_preflight"},
]
_RECOLOR_VARIANTS: list[JSONValue] = [
    *_FULL_EDIT_VARIANT,
    {"id": "operation_dsl", "mode": "operation_dsl"},
    {
        "id": "operation_dsl_preflight_no_dsl",
        "mode": "operation_dsl_preflight_no_dsl",
    },
    {
        "id": "operation_dsl_preflight_no_edit_mask",
        "mode": "operation_dsl_preflight_no_edit_mask",
    },
]
_RENDER_PROMPT = (
    "Photorealize the exact 3D layout. Preserve camera, room shell, openings, "
    "furniture count, object positions, object scale, and object colors; only "
    "improve realism, material finish, selected style, selected lighting, and "
    "selected scenery through existing windows."
)
_PRESET_SELECTION: dict[str, JSONValue] = {
    "style": "japandi",
    "lighting": "daylight_6",
    "scenery": "garden",
}
_CREATE_METRICS: list[JSONValue] = [
    "blank",
    "layout_fidelity",
    "layout_gate_score",
    "layout_edge_structure",
    "layout_depth_structure",
    "layout_dino_patch_similarity",
    "layout_dino_object_similarity",
    "structural_layout_score",
    "detector_layout",
    "detector_spatial_relations",
    "preset_fidelity",
]


@dataclass(frozen=True)
class ProfileConfig:
    layout_slug: str
    output_dir: Path
    recolor_object_id: str
    recolor_name: str
    recolor_color: str
    replace_object_id: str
    replace_name: str
    replace_description: str
    remove_object_id: str
    remove_name: str
    add_bbox_norm: Mapping[str, float]


_PROFILES: Mapping[Profile, ProfileConfig] = {
    "compact_reading": ProfileConfig(
        layout_slug="tknt_new_s_o_simple_living_02_compact_reading_tknt_full",
        output_dir=_RUNS_ROOT / "benchmark_1case_compact_reading_lean",
        recolor_object_id="sofa",
        recolor_name="sofa",
        recolor_color="muted forest green fabric",
        replace_object_id="tv_console",
        replace_name="TV console",
        replace_description="a slim walnut media console",
        remove_object_id="floor_lamp",
        remove_name="floor lamp",
        add_bbox_norm={"x": 0.75, "y": 0.48, "width": 0.08, "height": 0.24},
    ),
    "guest_vanity": ProfileConfig(
        layout_slug="tknt_new_s_o_simple_bedroom_04_guest_vanity_tknt_full",
        output_dir=_RUNS_ROOT / "benchmark_1case_guest_vanity_lean",
        recolor_object_id="wardrobe",
        recolor_name="wardrobe",
        recolor_color="matte sage green",
        replace_object_id="chair",
        replace_name="desk chair",
        replace_description="a compact upholstered desk chair",
        remove_object_id="desk_lamp_001",
        remove_name="desk lamp",
        add_bbox_norm={"x": 0.68, "y": 0.36, "width": 0.08, "height": 0.22},
    ),
}


def build_manifest(profile: Profile) -> dict[str, JSONValue]:
    config = _PROFILES[profile]
    case_dir = _CASES_ROOT / config.layout_slug
    snapshot_payload = case_dir / "snapshot_payload.json"
    snapshot_image = case_dir / "snapshot.png"
    payload = _read_mapping(snapshot_payload)
    _validate_add_region(payload, config.add_bbox_norm)
    create_case_id = f"{config.layout_slug}__create"
    edit_source_variant: dict[str, JSONValue] = {
        "case_id": create_case_id,
        "variant": "contract_objects_preflight_locked_geometry",
    }
    return {
        "defaults": {
            "metrics": _CREATE_METRICS,
            "preset_selection": _PRESET_SELECTION,
            "thresholds": {
                "max_layout_aspect_delta": 0.015,
                "min_layout_fidelity_score": 0.82,
                "min_layout_gate_score": 0.65,
                "min_layout_edge_structure_score": 0.35,
                "min_layout_depth_structure_score": 0.45,
                "min_layout_dino_patch_similarity": 0.50,
                "min_structural_layout_score": 0.58,
                "min_preset_fidelity_score": 0.75,
                "max_edit_outside_change_ratio": 0.08,
                "edit_bbox_margin_ratio": 0.015,
            },
            "align_mode": "resize_to_reference",
        },
        "output_dir": str(config.output_dir),
        "cases": [
            {
                "id": create_case_id,
                "snapshot_payload": str(snapshot_payload),
                "snapshot_image": str(snapshot_image),
                "render_prompt": _RENDER_PROMPT,
                "variants": _CREATE_VARIANTS,
            },
            _edit_case(
                config=config,
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
                suffix="edit_recolor",
                operation=_operation(
                    object_id=config.recolor_object_id,
                    object_name=config.recolor_name,
                    operation_type="recolor",
                    prompt=(
                        f"Change only the {config.recolor_name} to "
                        f"{config.recolor_color}; preserve its silhouette, size, "
                        "position, and every other object."
                    ),
                    bbox_norm=_object_bbox_norm(payload, config.recolor_object_id),
                    target_color=config.recolor_color,
                ),
                variants=_RECOLOR_VARIANTS,
                edit_source_variant=edit_source_variant,
            ),
            _edit_case(
                config=config,
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
                suffix="edit_replace",
                operation=_operation(
                    object_id=config.replace_object_id,
                    object_name=config.replace_name,
                    operation_type="replace",
                    prompt=(
                        f"Replace only the {config.replace_name} with "
                        f"{config.replace_description}, preserving the same bbox, "
                        "placement, scale, perspective, and occlusion."
                    ),
                    bbox_norm=_object_bbox_norm(payload, config.replace_object_id),
                ),
                variants=_FULL_EDIT_VARIANT,
                edit_source_variant=edit_source_variant,
            ),
            _edit_case(
                config=config,
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
                suffix="edit_add",
                operation=_operation(
                    object_id="add_region_floor_plant",
                    object_name="tall indoor plant",
                    operation_type="add",
                    prompt=(
                        "Add one slim tall indoor plant inside this region only; "
                        "do not move or resize any existing furniture."
                    ),
                    bbox_norm=config.add_bbox_norm,
                ),
                variants=_FULL_EDIT_VARIANT,
                edit_source_variant=edit_source_variant,
            ),
            _edit_case(
                config=config,
                snapshot_payload=snapshot_payload,
                snapshot_image=snapshot_image,
                suffix="edit_remove",
                operation=_operation(
                    object_id=config.remove_object_id,
                    object_name=config.remove_name,
                    operation_type="remove",
                    prompt=(
                        f"Remove only the {config.remove_name} and naturally "
                        "inpaint the background behind it; preserve every other object."
                    ),
                    bbox_norm=_object_bbox_norm(payload, config.remove_object_id),
                ),
                variants=_FULL_EDIT_VARIANT,
                edit_source_variant=edit_source_variant,
            ),
        ],
    }


def _edit_case(
    *,
    config: ProfileConfig,
    snapshot_payload: Path,
    snapshot_image: Path,
    suffix: str,
    operation: Mapping[str, JSONValue],
    variants: list[JSONValue],
    edit_source_variant: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    return {
        "id": f"{config.layout_slug}__{suffix}",
        "snapshot_payload": str(snapshot_payload),
        "snapshot_image": str(snapshot_image),
        "render_prompt": _RENDER_PROMPT,
        "edit_prompt": str(operation["prompt"]),
        "edit_source_variant": dict(edit_source_variant),
        "edit_operations": [dict(operation)],
        "variants": variants,
    }


def _operation(
    *,
    object_id: str,
    object_name: str,
    operation_type: str,
    prompt: str,
    bbox_norm: Mapping[str, float],
    target_color: str | None = None,
) -> dict[str, JSONValue]:
    operation: dict[str, JSONValue] = {
        "object_id": object_id,
        "object_name": object_name,
        "operation_type": operation_type,
        "prompt": prompt,
        "bbox_norm": dict(bbox_norm),
    }
    if target_color is not None:
        operation["target_color"] = target_color
    return operation


def _object_bbox_norm(
    payload: Mapping[str, object],
    object_id: str,
) -> dict[str, float]:
    canvas = _mapping(payload.get("canvas"))
    width = _positive_float(canvas.get("widthPx"), field="canvas.widthPx")
    height = _positive_float(canvas.get("heightPx"), field="canvas.heightPx")
    for item in _sequence(payload.get("visibleObjects")):
        visible_object = _mapping(item)
        if visible_object.get("id") != object_id:
            continue
        bbox = _mapping(visible_object.get("screenBboxPx"))
        return {
            "x": round(_positive_or_zero_float(bbox.get("minX")) / width, 4),
            "y": round(_positive_or_zero_float(bbox.get("minY")) / height, 4),
            "width": round(
                _positive_float(bbox.get("width"), field="bbox.width") / width, 4
            ),
            "height": round(
                _positive_float(bbox.get("height"), field="bbox.height") / height,
                4,
            ),
        }
    raise ValueError(f"Visible object {object_id!r} was not found in snapshot payload.")


def _validate_add_region(
    payload: Mapping[str, object],
    add_bbox_norm: Mapping[str, float],
) -> None:
    for item in _sequence(payload.get("visibleObjects")):
        visible_object = _mapping(item)
        object_id = str(visible_object.get("id") or "")
        if not object_id:
            continue
        object_bbox = _object_bbox_norm(payload, object_id)
        if _intersection_area(add_bbox_norm, object_bbox) > 0.001:
            raise ValueError(
                f"Add region overlaps visible object {object_id!r}; choose a free region."
            )


def _intersection_area(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    min_x = max(left["x"], right["x"])
    min_y = max(left["y"], right["y"])
    max_x = min(left["x"] + left["width"], right["x"] + right["width"])
    max_y = min(left["y"] + left["height"], right["y"] + right["height"])
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


def _read_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _sequence(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _positive_float(value: object, *, field: str) -> float:
    parsed = _positive_or_zero_float(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive number for {field}.")
    return parsed


def _positive_or_zero_float(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected numeric snapshot bbox value.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build lean stabilization benchmark manifests from current snapshots."
    )
    parser.add_argument("--profile", choices=tuple(_PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = cast(Profile, args.profile)
    output = cast(Path, args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_manifest(profile), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote stabilization manifest: %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
