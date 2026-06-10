from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import httpx
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    UnidentifiedImageError,
)

from .io import encode_image_to_data_url, write_data_url_image, write_json_file
from .types import JSONValue

logger = logging.getLogger(__name__)

_MODEL_ENV = "TKNT_IMAGE_FLOW_REPAIR_MODEL"
_BASE_URL_ENV = "TKNT_IMAGE_FLOW_REPAIR_OPENAI_BASE_URL"
_PREFLIGHT_BASE_URL_ENV = "TKNT_IMAGE_FLOW_PREFLIGHT_OPENAI_BASE_URL"
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_IMAGE_API_KEY_ENV = "OPENAI_IMAGE_API_KEY"
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_SOURCE_VARIANT = "single_edit_edge_silhouette_guided"
_DEFAULT_OUTPUT_VARIANT = "single_edit_edge_silhouette_guided_llm_repair"
_ENHANCED_GEOMETRY_GUIDE_NAME = "enhanced_geometry_guide.png"
_PLAN_FILENAME = "llm_repair_plan.json"
_RESULT_FILENAME = "tknt_snapshot_result.json"
_SUMMARY_FILENAME = "semantic_repair_summary.json"
_LAYOUT_GUARD_BORDER_RATIO = 0.035
_MAX_GUIDE_OBJECT_ANCHORS = 24

RepairCaseStatus = Literal[
    "generated",
    "reused_existing",
    "skipped_no_repair_needed",
    "guide_only",
    "failed",
]


@dataclass(frozen=True)
class SemanticRepairInput:
    case_id: str
    source_variant_dir: Path
    output_variant_dir: Path
    source_image: Path
    pass1_render_image: Path
    source_payload: Path
    semantic_guide_image: Path | None


@dataclass(frozen=True)
class SemanticRepairResult:
    case_id: str
    status: RepairCaseStatus
    output_image: Path | None
    plan_path: Path | None
    metadata_path: Path | None
    error_message: str | None = None

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "output_image": str(self.output_image) if self.output_image else None,
            "plan_path": str(self.plan_path) if self.plan_path else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "error_message": self.error_message,
        }


def run_semantic_repair(
    *,
    run_dir: Path,
    output_dir: Path,
    source_variant: str = _DEFAULT_SOURCE_VARIANT,
    output_variant: str = _DEFAULT_OUTPUT_VARIANT,
    case_ids: Sequence[str] = (),
    resume: bool = False,
    plan_only: bool = False,
    force_repair: bool = False,
    overwrite: bool = False,
    refresh_plan: bool = False,
    enhanced_geometry_guide: bool = False,
    guide_only: bool = False,
) -> list[SemanticRepairResult]:
    inputs = discover_semantic_repair_inputs(
        run_dir=run_dir,
        output_dir=output_dir,
        source_variant=source_variant,
        output_variant=output_variant,
        case_ids=case_ids,
    )
    results: list[SemanticRepairResult] = []
    failed = False
    for repair_input in inputs:
        try:
            results.append(
                run_semantic_repair_case(
                    repair_input,
                    resume=resume,
                    plan_only=plan_only,
                    force_repair=force_repair,
                    overwrite=overwrite,
                    refresh_plan=refresh_plan,
                    enhanced_geometry_guide=enhanced_geometry_guide,
                    guide_only=guide_only,
                )
            )
        except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as exc:
            failed = True
            logger.exception("Semantic repair failed for case %s", repair_input.case_id)
            results.append(
                SemanticRepairResult(
                    case_id=repair_input.case_id,
                    status="failed",
                    output_image=None,
                    plan_path=None,
                    metadata_path=None,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
    _write_summary(
        output_dir=output_dir,
        results=results,
        run_dir=run_dir,
        source_variant=source_variant,
        output_variant=output_variant,
        plan_only=plan_only,
        force_repair=force_repair,
        overwrite=overwrite,
        refresh_plan=refresh_plan,
        enhanced_geometry_guide=enhanced_geometry_guide,
        guide_only=guide_only,
    )
    if failed:
        logger.warning("Semantic repair completed with failures.")
    return results


def discover_semantic_repair_inputs(
    *,
    run_dir: Path,
    output_dir: Path,
    source_variant: str = _DEFAULT_SOURCE_VARIANT,
    output_variant: str = _DEFAULT_OUTPUT_VARIANT,
    case_ids: Sequence[str] = (),
) -> list[SemanticRepairInput]:
    selected_case_ids = set(case_ids)
    root = run_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Semantic repair run directory does not exist: {root}")
    discovered: list[SemanticRepairInput] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected_case_ids and case_dir.name not in selected_case_ids:
            continue
        variant_dir = case_dir / source_variant
        source_image = variant_dir / "source_3d.png"
        pass1_render = variant_dir / "render.png"
        source_payload = variant_dir / "source_snapshot_payload.json"
        if not (
            source_image.is_file()
            and pass1_render.is_file()
            and source_payload.is_file()
        ):
            continue
        semantic_guide = variant_dir / "room_shell_semantic_guide.png"
        discovered.append(
            SemanticRepairInput(
                case_id=case_dir.name,
                source_variant_dir=variant_dir,
                output_variant_dir=(
                    output_dir.expanduser().resolve() / case_dir.name / output_variant
                ),
                source_image=source_image,
                pass1_render_image=pass1_render,
                source_payload=source_payload,
                semantic_guide_image=semantic_guide
                if semantic_guide.is_file()
                else None,
            )
        )
    if selected_case_ids:
        found = {item.case_id for item in discovered}
        missing = sorted(selected_case_ids - found)
        if missing:
            raise ValueError(
                "No reusable semantic-pass output found for case(s): "
                + ", ".join(missing)
            )
    if not discovered:
        raise ValueError(
            f"No reusable {source_variant} outputs found under semantic repair run: {root}"
        )
    return discovered


def run_semantic_repair_case(
    repair_input: SemanticRepairInput,
    *,
    resume: bool = False,
    plan_only: bool = False,
    force_repair: bool = False,
    overwrite: bool = False,
    refresh_plan: bool = False,
    enhanced_geometry_guide: bool = False,
    guide_only: bool = False,
) -> SemanticRepairResult:
    output_dir = repair_input.output_variant_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_render = output_dir / "render.png"
    metadata_path = output_dir / _RESULT_FILENAME
    plan_path = output_dir / _PLAN_FILENAME
    if (
        resume
        and not overwrite
        and output_render.is_file()
        and metadata_path.is_file()
        and not plan_only
    ):
        return SemanticRepairResult(
            case_id=repair_input.case_id,
            status="reused_existing",
            output_image=output_render,
            plan_path=plan_path if plan_path.is_file() else None,
            metadata_path=metadata_path,
        )

    copied_paths, guide_metadata = _copy_repair_inputs(
        repair_input,
        enhanced_geometry_guide=enhanced_geometry_guide,
    )
    source_payload = _read_json_mapping(copied_paths["source_payload"])
    if guide_only:
        _write_result_metadata(
            metadata_path=metadata_path,
            repair_input=repair_input,
            copied_paths=copied_paths,
            guide_metadata=guide_metadata,
            plan={
                "repair_needed": True,
                "overall_status": "guide_only",
                "issues": [],
                "repair_prompt": "",
                "model": _model(),
                "planner_elapsed_ms": None,
                "usage": {},
            },
            render_result=None,
            output_image=None,
            status="guide_only",
            elapsed_ms=None,
        )
        return SemanticRepairResult(
            case_id=repair_input.case_id,
            status="guide_only",
            output_image=None,
            plan_path=None,
            metadata_path=metadata_path,
        )
    plan = _load_or_request_repair_plan(
        repair_input=repair_input,
        plan_path=plan_path,
        source_payload=source_payload,
        resume=resume,
        refresh_plan=refresh_plan,
        planner_guide_image=copied_paths.get("annotated_reference_image"),
    )
    if plan_only:
        _write_result_metadata(
            metadata_path=metadata_path,
            repair_input=repair_input,
            copied_paths=copied_paths,
            guide_metadata=guide_metadata,
            plan=plan,
            render_result=None,
            output_image=None,
            status="generated",
            elapsed_ms=None,
        )
        return SemanticRepairResult(
            case_id=repair_input.case_id,
            status="generated",
            output_image=None,
            plan_path=plan_path,
            metadata_path=metadata_path,
        )

    if not force_repair and not bool(plan.get("repair_needed", True)):
        shutil.copy2(copied_paths["pass1_render_image"], output_render)
        _write_result_metadata(
            metadata_path=metadata_path,
            repair_input=repair_input,
            copied_paths=copied_paths,
            guide_metadata=guide_metadata,
            plan=plan,
            render_result=None,
            output_image=output_render,
            status="skipped_no_repair_needed",
            elapsed_ms=None,
        )
        return SemanticRepairResult(
            case_id=repair_input.case_id,
            status="skipped_no_repair_needed",
            output_image=output_render,
            plan_path=plan_path,
            metadata_path=metadata_path,
        )

    started_at = perf_counter()
    render_result = _render_repair(
        repair_input=repair_input,
        copied_paths=copied_paths,
        plan=plan,
        force_repair=force_repair,
    )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    output_data_url = _extract_result_data_url(render_result)
    write_data_url_image(output_data_url, output_render)
    _write_result_metadata(
        metadata_path=metadata_path,
        repair_input=repair_input,
        copied_paths=copied_paths,
        guide_metadata=guide_metadata,
        plan=plan,
        render_result=render_result,
        output_image=output_render,
        status="generated",
        elapsed_ms=elapsed_ms,
    )
    return SemanticRepairResult(
        case_id=repair_input.case_id,
        status="generated",
        output_image=output_render,
        plan_path=plan_path,
        metadata_path=metadata_path,
    )


def _copy_repair_inputs(
    repair_input: SemanticRepairInput,
    *,
    enhanced_geometry_guide: bool,
) -> tuple[dict[str, Path], dict[str, JSONValue]]:
    output_dir = repair_input.output_variant_dir
    copied: dict[str, Path] = {}
    guide_metadata: dict[str, JSONValue] = {
        "enhanced_geometry_guide_enabled": enhanced_geometry_guide,
        "enhanced_geometry_guide_image": None,
        "annotated_reference_image": None,
    }
    copies = {
        "source_image": repair_input.source_image,
        "pass1_render_image": repair_input.pass1_render_image,
        "source_payload": repair_input.source_payload,
    }
    if repair_input.semantic_guide_image is not None:
        copies["semantic_guide_image"] = repair_input.semantic_guide_image
    for key, source in copies.items():
        destination = output_dir / _copied_input_name(key, source)
        shutil.copy2(source, destination)
        copied[key] = destination
    if enhanced_geometry_guide:
        source_payload = _read_json_mapping(copied["source_payload"])
        enhanced_path = output_dir / _ENHANCED_GEOMETRY_GUIDE_NAME
        enhanced_metadata = _build_enhanced_geometry_guide(
            source_image_path=copied["source_image"],
            source_payload=source_payload,
            output_path=enhanced_path,
        )
        copied["enhanced_geometry_guide_image"] = enhanced_path
        copied["annotated_reference_image"] = enhanced_path
        guide_metadata = {
            **guide_metadata,
            **enhanced_metadata,
            "enhanced_geometry_guide_image": str(enhanced_path),
            "annotated_reference_image": str(enhanced_path),
        }
    elif copied.get("semantic_guide_image") is not None:
        copied["annotated_reference_image"] = copied["semantic_guide_image"]
        guide_metadata = {
            **guide_metadata,
            "annotated_reference_image": str(copied["semantic_guide_image"]),
        }
    return copied, guide_metadata


def _copied_input_name(key: str, source: Path) -> str:
    if key == "source_image":
        return "source_3d.png"
    if key == "pass1_render_image":
        return "semantic_pass_render.png"
    if key == "source_payload":
        return "source_snapshot_payload.json"
    if key == "semantic_guide_image":
        return source.name
    return source.name


def _build_enhanced_geometry_guide(
    *,
    source_image_path: Path,
    source_payload: Mapping[str, object],
    output_path: Path,
) -> dict[str, JSONValue]:
    try:
        source = Image.open(source_image_path).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(
            f"Unable to decode source image for enhanced geometry guide: {source_image_path}"
        ) from exc

    enhanced = ImageOps.autocontrast(source, cutoff=1)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.22)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.65)
    enhanced = enhanced.filter(
        ImageFilter.UnsharpMask(radius=1.15, percent=135, threshold=3)
    )
    guide = enhanced.convert("RGBA")
    image_width, image_height = guide.size
    overlay = Image.new("RGBA", guide.size, (0, 0, 0, 0))

    edge_mask = (
        ImageOps.autocontrast(source.convert("L").filter(ImageFilter.FIND_EDGES))
        .filter(ImageFilter.MaxFilter(3))
        .point(lambda value: 190 if value >= 30 else 0)
    )
    edge_overlay = Image.new("RGBA", guide.size, (255, 116, 0, 0))
    edge_overlay.putalpha(edge_mask)
    overlay = Image.alpha_composite(overlay, edge_overlay)

    draw = ImageDraw.Draw(overlay)
    stroke_width = max(2, round(min(image_width, image_height) / 420))
    border = max(8, round(min(image_width, image_height) * _LAYOUT_GUARD_BORDER_RATIO))
    _draw_crop_guard(
        draw,
        image_size=guide.size,
        border=border,
        stroke_width=stroke_width + 1,
    )
    opening_count = _draw_opening_anchors(
        draw,
        source_payload=source_payload,
        stroke_width=stroke_width + 1,
    )
    clean_count, rejected_count = _draw_object_anchors(
        draw,
        source_payload=source_payload,
        image_size=guide.size,
        stroke_width=stroke_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(guide, overlay).convert("RGB").save(output_path)
    return {
        "enhanced_geometry_guide_enabled": True,
        "guide_type": "deterministic_enhanced_geometry",
        "edge_method": "pil_find_edges_autocontrast_maxfilter_threshold",
        "edge_threshold": 30,
        "contrast_factor": 1.22,
        "sharpness_factor": 1.65,
        "unsharp_mask": {"radius": 1.15, "percent": 135, "threshold": 3},
        "crop_guard_border_px": border,
        "opening_anchor_count": opening_count,
        "object_anchor_count": clean_count,
        "rejected_object_anchor_count": rejected_count,
        "guide_semantics": (
            "same-size deterministic structural guide; orange edges/crop guard, "
            "blue openings, green clean object anchors, red rejected noisy anchors"
        ),
    }


def _draw_crop_guard(
    draw: ImageDraw.ImageDraw,
    *,
    image_size: tuple[int, int],
    border: int,
    stroke_width: int,
) -> None:
    image_width, image_height = image_size
    draw.rectangle(
        [0, 0, image_width - 1, image_height - 1],
        outline=(255, 132, 0, 245),
        width=stroke_width,
    )
    for box in (
        [0, 0, border, image_height],
        [image_width - border, 0, image_width, image_height],
        [0, 0, image_width, border],
        [0, image_height - border, image_width, image_height],
    ):
        draw.rectangle(box, fill=(255, 132, 0, 28))


def _draw_opening_anchors(
    draw: ImageDraw.ImageDraw,
    *,
    source_payload: Mapping[str, object],
    stroke_width: int,
) -> int:
    count = 0
    for bbox in _opening_bboxes(source_payload):
        _draw_outline_bbox(
            draw,
            bbox,
            outline=(0, 112, 255, 240),
            fill=(0, 112, 255, 28),
            stroke_width=stroke_width,
        )
        count += 1
    return count


def _draw_object_anchors(
    draw: ImageDraw.ImageDraw,
    *,
    source_payload: Mapping[str, object],
    image_size: tuple[int, int],
    stroke_width: int,
) -> tuple[int, int]:
    anchors = _visible_object_anchor_bboxes(source_payload, image_size=image_size)
    clean_count = 0
    rejected_count = 0
    for anchor in anchors[:_MAX_GUIDE_OBJECT_ANCHORS]:
        color = (34, 197, 94, 230)
        fill = (34, 197, 94, 18)
        if bool(anchor.get("rejected")):
            color = (220, 38, 38, 220)
            fill = None
            rejected_count += 1
        else:
            clean_count += 1
        bbox = _json_float_mapping(anchor.get("screen_bbox_px"))
        if not bbox:
            continue
        _draw_outline_bbox(
            draw,
            bbox,
            outline=color,
            fill=fill,
            stroke_width=stroke_width,
        )
    return clean_count, rejected_count


def _opening_bboxes(source_payload: Mapping[str, object]) -> list[dict[str, float]]:
    boxes: list[dict[str, float]] = []
    for container in (
        _mapping(source_payload.get("openings")),
        _mapping(_mapping(source_payload.get("room")).get("openings")),
    ):
        for key in ("windows", "doors"):
            for item in _sequence(container.get(key)):
                bbox = _screen_bbox(_mapping(item).get("screenBboxPx"))
                if bbox is not None:
                    boxes.append(bbox)
    return boxes


def _visible_object_anchor_bboxes(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> list[dict[str, JSONValue]]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    image_width, image_height = image_size
    anchors: list[dict[str, JSONValue]] = []
    for item in visible_objects:
        if not isinstance(item, Mapping):
            continue
        bbox = _screen_bbox(item.get("screenBboxPx"))
        if bbox is None:
            continue
        metrics = _normalized_bbox_metrics(
            bbox,
            image_width=image_width,
            image_height=image_height,
        )
        area = metrics["area"]
        edge_touch_count = int(metrics["edge_touch_count"])
        rejected = area > 0.42 or edge_touch_count >= 3
        anchors.append(
            {
                "screen_bbox_px": bbox,
                "area": area,
                "rejected": rejected,
            }
        )
    anchors.sort(key=lambda value: float(value["area"]), reverse=True)
    return anchors


def _draw_outline_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int] | None,
    stroke_width: int,
) -> None:
    min_x = round(bbox["minX"])
    min_y = round(bbox["minY"])
    max_x = round(bbox["maxX"])
    max_y = round(bbox["maxY"])
    if max_x <= min_x or max_y <= min_y:
        return
    box = [min_x, min_y, max_x, max_y]
    if fill is not None:
        draw.rectangle(box, fill=fill)
    draw.rectangle(box, outline=outline, width=stroke_width)


def _screen_bbox(value: object) -> dict[str, float] | None:
    payload = _mapping(value)
    min_x = _float_value(payload.get("minX"))
    min_y = _float_value(payload.get("minY"))
    if min_x is None:
        min_x = _float_value(payload.get("x"))
    if min_y is None:
        min_y = _float_value(payload.get("y"))
    width = _float_value(payload.get("width"))
    height = _float_value(payload.get("height"))
    max_x = _float_value(payload.get("maxX"))
    max_y = _float_value(payload.get("maxY"))
    if width is None and max_x is not None and min_x is not None:
        width = max_x - min_x
    if height is None and max_y is not None and min_y is not None:
        height = max_y - min_y
    if min_x is None or min_y is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": min_x + width,
        "maxY": min_y + height,
        "width": width,
        "height": height,
    }


def _normalized_bbox_metrics(
    bbox: Mapping[str, float],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    width = max(0.0, bbox["width"] / max(1, image_width))
    height = max(0.0, bbox["height"] / max(1, image_height))
    x = max(0.0, min(1.0, bbox["minX"] / max(1, image_width)))
    y = max(0.0, min(1.0, bbox["minY"] / max(1, image_height)))
    edge_touch_count = float(
        sum(
            (
                x <= 0.01,
                y <= 0.01,
                x + width >= 0.99,
                y + height >= 0.99,
            )
        )
    )
    return {
        "area": width * height,
        "edge_touch_count": edge_touch_count,
    }


def _json_float_mapping(value: object) -> dict[str, float]:
    payload = _mapping(value)
    parsed: dict[str, float] = {}
    for key in ("minX", "minY", "maxX", "maxY", "width", "height"):
        number = _float_value(payload.get(key))
        if number is not None:
            parsed[key] = number
    return parsed


def _load_or_request_repair_plan(
    *,
    repair_input: SemanticRepairInput,
    plan_path: Path,
    source_payload: Mapping[str, object],
    resume: bool,
    refresh_plan: bool,
    planner_guide_image: Path | None,
) -> dict[str, JSONValue]:
    if resume and plan_path.is_file() and not refresh_plan:
        return _normalize_repair_plan(_read_json_mapping(plan_path))
    plan = _request_repair_plan(
        repair_input=repair_input,
        source_payload=source_payload,
        planner_guide_image=planner_guide_image,
    )
    write_json_file(plan_path, plan)
    return plan


def _request_repair_plan(
    *,
    repair_input: SemanticRepairInput,
    source_payload: Mapping[str, object],
    planner_guide_image: Path | None,
) -> dict[str, JSONValue]:
    api_key = _api_key()
    if not api_key:
        raise ValueError(
            "Semantic repair requires OPENAI_API_KEY or OPENAI_IMAGE_API_KEY."
        )
    model = _model()
    prompt = _repair_planner_prompt(source_payload=source_payload)
    content: list[dict[str, object]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_to_data_url(repair_input.source_image)},
        },
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image_to_data_url(repair_input.pass1_render_image)
            },
        },
    ]
    if planner_guide_image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(planner_guide_image)},
            }
        )

    endpoint = _base_url().rstrip("/") + "/chat/completions"
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict interior layout repair planner. Return only JSON."
                ),
            },
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }
    started_at = perf_counter()
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    response.raise_for_status()
    payload = _read_json_mapping_from_object(response.json())
    report = _extract_chat_json(payload)
    usage = _json_mapping(payload.get("usage"))
    return _normalize_repair_plan(
        {
            **report,
            "model": model,
            "planner_elapsed_ms": elapsed_ms,
            "usage": usage,
        }
    )


def _repair_planner_prompt(*, source_payload: Mapping[str, object]) -> str:
    visible_objects = _visible_object_summary(source_payload)
    room_summary = _room_shell_summary(source_payload)
    optional_image3 = (
        "Image 3, if present, is a deterministic structural guide generated from "
        "Image 1. It may contain sharpened edges, crop guards, opening marks, and "
        "object anchors. Use it only to read geometry; colored marks are not scene "
        "content."
    )
    return (
        "Compare the images for layout repair only.\n\n"
        "Image 1: clean 3D source. This is the ground truth for camera, crop, wall "
        "planes, corner seams, openings, object count, object positions, scale, "
        "silhouettes, depth order, and occlusion.\n"
        "Image 2: current photoreal render produced by the semantic guided edit. "
        "This is the image that may need targeted repair.\n"
        f"{optional_image3}\n\n"
        "Ignore harmless material, texture, lighting, and style differences. Flag "
        "only geometry/layout mistakes: missing walls/corners, invented openings, "
        "wrong crop, wrong wall plane, moved/missing/added furniture, bad object "
        "scale, bad depth spacing, distorted TV/media cluster, foreground object "
        "drift, or mistaking a shadow for a physical object.\n\n"
        "Be skeptical. Do not mark no_repair_needed just because Image 2 is a "
        "plausible room. It must match Image 1's visible room shell and visible "
        "object layout. Pay special attention to near crop-edge walls, subtle wall "
        "planes at the left/right image edge, wall seams that look like shadows, "
        "TV/media/speaker clusters, beds/sofas/rugs, foreground cropped objects, "
        "and any object whose bbox is small but layout-critical.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "repair_needed": true,\n'
        '  "overall_status": "needs_repair",\n'
        '  "issues": [\n'
        "    {\n"
        '      "category": "room_shell|opening|object_inventory|object_position|object_scale|depth_spacing|shadow_confusion|crop|other",\n'
        '      "severity": "minor|major|critical",\n'
        '      "source_evidence": "what Image 1 shows",\n'
        '      "render_problem": "what Image 2 got wrong",\n'
        '      "repair_instruction": "one targeted instruction for image edit"\n'
        "    }\n"
        "  ],\n"
        '  "repair_prompt": "concise image-edit prompt that fixes only the listed issues"\n'
        "}\n\n"
        "The repair_prompt must be narrow. It must say to edit Image 2 while using "
        "Image 1 as geometry reference, preserve Image 2's photoreal style and all "
        "correct regions, and not regenerate or redesign the room. If no layout "
        "repair is needed, set repair_needed=false only after explicitly checking "
        "room shell, openings, inventory, object positions, object scale, depth "
        "spacing, and crop-edge walls.\n\n"
        f"Room/opening summary from metadata:\n{room_summary}\n\n"
        f"Visible objects from metadata:\n{visible_objects}"
    )


def _render_repair(
    *,
    repair_input: SemanticRepairInput,
    copied_paths: Mapping[str, Path],
    plan: Mapping[str, JSONValue],
    force_repair: bool,
) -> Mapping[str, object]:
    from dataclasses import replace

    from pipeline.snapshot_image_renderer import (
        SnapshotImageRenderConfig,
        render_snapshot_image_from_path,
    )

    annotated_reference_image = copied_paths.get("annotated_reference_image")
    config = replace(
        SnapshotImageRenderConfig.from_env(),
        include_annotated_reference=annotated_reference_image is not None,
        include_geometry_references=False,
        include_locked_geometry_references=False,
        include_preset_reference_images=False,
        simple_prompt_mode=False,
        text_contract_mode=False,
        include_edit_operation_dsl=False,
        strict_layout_mode=True,
    )
    repair_prompt = _final_repair_prompt(plan, force_repair=force_repair)
    return render_snapshot_image_from_path(
        copied_paths["source_payload"],
        snapshot_image_data_url=encode_image_to_data_url(copied_paths["source_image"]),
        user_prompt=repair_prompt,
        annotated_reference_image_data_url=encode_image_to_data_url(
            annotated_reference_image
        )
        if annotated_reference_image is not None
        else None,
        edit_source_image_data_url=encode_image_to_data_url(
            copied_paths["pass1_render_image"]
        ),
        scene_reference_mode="scene_reference_camera_transfer",
        render_mode="edit",
        preset_selection={},
        config=config,
        trace_id=f"{repair_input.case_id}:semantic_llm_repair",
    )


def _final_repair_prompt(
    plan: Mapping[str, JSONValue],
    *,
    force_repair: bool = False,
) -> str:
    repair_prompt = _string(plan.get("repair_prompt"))
    issue_lines = _issue_prompt_lines(plan.get("issues"))
    if not repair_prompt:
        repair_prompt = "\n".join(issue_lines) if issue_lines else ""
    if force_repair and not repair_prompt:
        repair_prompt = (
            "Conservatively compare the current render against the clean 3D source "
            "and correct any layout mismatch you can see: missing crop-edge wall "
            "planes, wall seams, invented openings, wrong object count, shifted or "
            "rescaled furniture, wrong depth spacing, distorted TV/media/speaker "
            "cluster, foreground cropped object drift, and shadow-vs-geometry "
            "confusion. If a region already matches the 3D source, keep it unchanged."
        )
    if not repair_prompt:
        repair_prompt = "Repair layout drift."
    return (
        "Targeted layout repair pass.\n"
        "Edit Image 1, the current photoreal render. Use Image 2, the clean 3D "
        "source, only as geometry/camera/layout truth. Preserve Image 1's correct "
        "photoreal materials, style, lighting, colors, and all correct regions. "
        "Do not regenerate the whole room. Do not add new openings, furniture, "
        "decor, people, labels, guide marks, or text. If a structural guide image "
        "is provided, use it only to locate wall seams, crop-edge wall planes, "
        "openings, object anchors, silhouettes, and depth/crop warnings; never "
        f"render its colors or lines. Fix only these layout problems:\n{repair_prompt}"
    )


def _issue_prompt_lines(value: object) -> list[str]:
    issues = value if isinstance(value, list) else []
    lines: list[str] = []
    for index, item in enumerate(issues, start=1):
        if not isinstance(item, Mapping):
            continue
        instruction = _string(item.get("repair_instruction"))
        if not instruction:
            continue
        category = _string(item.get("category"), fallback="layout")
        severity = _string(item.get("severity"), fallback="major")
        lines.append(f"{index}. [{severity} {category}] {instruction}")
    return lines


def _normalize_repair_plan(value: Mapping[str, object]) -> dict[str, JSONValue]:
    issues = _normalize_issues(value.get("issues"))
    repair_needed = _bool_value(value.get("repair_needed"), fallback=bool(issues))
    repair_prompt = _string(value.get("repair_prompt"))
    if repair_needed and not repair_prompt:
        repair_prompt = "\n".join(
            str(issue["repair_instruction"])
            for issue in issues
            if isinstance(issue.get("repair_instruction"), str)
        )
    return {
        "repair_needed": repair_needed,
        "overall_status": _string(
            value.get("overall_status"),
            fallback="needs_repair" if repair_needed else "no_repair_needed",
        ),
        "issues": issues,
        "repair_prompt": repair_prompt,
        "model": _string(value.get("model"), fallback=_model()),
        "planner_elapsed_ms": _float_value(value.get("planner_elapsed_ms")),
        "usage": _json_mapping(value.get("usage")),
    }


def _normalize_issues(value: object) -> list[dict[str, JSONValue]]:
    if not isinstance(value, list):
        return []
    issues: list[dict[str, JSONValue]] = []
    for item in value[:8]:
        if not isinstance(item, Mapping):
            continue
        repair_instruction = _string(item.get("repair_instruction"))
        if not repair_instruction:
            continue
        issues.append(
            {
                "category": _string(item.get("category"), fallback="other"),
                "severity": _string(item.get("severity"), fallback="major"),
                "source_evidence": _string(item.get("source_evidence")),
                "render_problem": _string(item.get("render_problem")),
                "repair_instruction": repair_instruction,
            }
        )
    return issues


def _write_result_metadata(
    *,
    metadata_path: Path,
    repair_input: SemanticRepairInput,
    copied_paths: Mapping[str, Path],
    guide_metadata: Mapping[str, JSONValue],
    plan: Mapping[str, JSONValue],
    render_result: Mapping[str, object] | None,
    output_image: Path | None,
    status: RepairCaseStatus,
    elapsed_ms: float | None,
) -> None:
    payload: dict[str, JSONValue] = {
        "case_id": repair_input.case_id,
        "variant_id": repair_input.output_variant_dir.name,
        "mode": "semantic_llm_repair",
        "provider_status": status,
        "source_variant_dir": str(repair_input.source_variant_dir),
        "source_image": str(copied_paths["source_image"]),
        "source_snapshot_payload": str(copied_paths["source_payload"]),
        "semantic_pass_render_image": str(copied_paths["pass1_render_image"]),
        "semantic_guide_image": str(copied_paths["semantic_guide_image"])
        if copied_paths.get("semantic_guide_image") is not None
        else None,
        "enhanced_geometry_guide_image": str(
            copied_paths["enhanced_geometry_guide_image"]
        )
        if copied_paths.get("enhanced_geometry_guide_image") is not None
        else None,
        "annotated_reference_image": str(copied_paths["annotated_reference_image"])
        if copied_paths.get("annotated_reference_image") is not None
        else None,
        "geometry_guide": dict(guide_metadata),
        "repair_plan": dict(plan),
        "render_image": str(output_image) if output_image is not None else None,
        "output_image": str(output_image) if output_image is not None else None,
        "repair_elapsed_ms": elapsed_ms,
        "render_result": _summarize_render_result(render_result),
    }
    write_json_file(metadata_path, payload)


def _write_summary(
    *,
    output_dir: Path,
    results: Sequence[SemanticRepairResult],
    run_dir: Path,
    source_variant: str,
    output_variant: str,
    plan_only: bool,
    force_repair: bool,
    overwrite: bool,
    refresh_plan: bool,
    enhanced_geometry_guide: bool,
    guide_only: bool,
) -> None:
    payload: dict[str, JSONValue] = {
        "run_dir": str(run_dir.expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "source_variant": source_variant,
        "output_variant": output_variant,
        "plan_only": plan_only,
        "force_repair": force_repair,
        "overwrite": overwrite,
        "refresh_plan": refresh_plan,
        "enhanced_geometry_guide": enhanced_geometry_guide,
        "guide_only": guide_only,
        "case_count": len(results),
        "generated_count": sum(1 for item in results if item.status == "generated"),
        "failed_count": sum(1 for item in results if item.status == "failed"),
        "results": [item.to_json() for item in results],
    }
    write_json_file(output_dir / _SUMMARY_FILENAME, payload)


def _extract_chat_json(payload: Mapping[str, object]) -> Mapping[str, object]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI chat response contained no choices.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("OpenAI chat choice is not an object.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("OpenAI chat choice has no message object.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI chat message has no JSON content.")
    parsed = json.loads(content)
    if not isinstance(parsed, Mapping):
        raise ValueError("OpenAI chat JSON content must be an object.")
    return cast(Mapping[str, object], parsed)


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    return _read_json_mapping_from_object(json.loads(path.read_text(encoding="utf-8")))


def _read_json_mapping_from_object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected JSON object.")
    return cast(Mapping[str, object], value)


def _extract_result_data_url(result: Mapping[str, object]) -> str:
    image = result.get("image")
    if isinstance(image, Mapping):
        data_url = image.get("data_url")
        if isinstance(data_url, str) and data_url.startswith("data:image/"):
            return data_url
    data_url = result.get("data_url")
    if isinstance(data_url, str) and data_url.startswith("data:image/"):
        return data_url
    raise ValueError("Render result did not contain an image data URL.")


def _summarize_render_result(result: Mapping[str, object] | None) -> JSONValue:
    if result is None:
        return None
    image = result.get("image")
    image_payload = image if isinstance(image, Mapping) else result
    return {
        "request": _json_safe(result.get("request"))
        if isinstance(result.get("request"), Mapping)
        else None,
        "metadata": _json_safe(result.get("metadata"))
        if isinstance(result.get("metadata"), Mapping)
        else None,
        "image": _redact_image_payload(image_payload),
    }


def _redact_image_payload(value: object) -> JSONValue:
    if not isinstance(value, Mapping):
        return None
    redacted: dict[str, JSONValue] = {}
    for key, item in value.items():
        if key in {"data_url", "image_base64", "b64_json"}:
            if isinstance(item, str):
                redacted[key] = f"<redacted {len(item)} chars>"
            continue
        redacted[str(key)] = _json_safe(item)
    return redacted


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_safe(item) for item in value]
    return str(value)


def _visible_object_summary(source_payload: Mapping[str, object]) -> str:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return "No visibleObjects metadata found."
    lines: list[str] = []
    for item in visible_objects[:24]:
        if not isinstance(item, Mapping):
            continue
        object_id = _string(item.get("id"), fallback="unknown")
        object_type = _string(item.get("type"), fallback="object")
        bbox = _bbox_text(item.get("screenBboxPx"))
        clipped = bool(item.get("clipped", False))
        clip_text = ", clipped" if clipped else ""
        lines.append(f"- {object_id} ({object_type}){clip_text}{bbox}")
    return "\n".join(lines) if lines else "No visibleObjects metadata found."


def _room_shell_summary(source_payload: Mapping[str, object]) -> str:
    parts: list[str] = []
    openings = source_payload.get("openings")
    if isinstance(openings, Mapping):
        doors = openings.get("doors")
        windows = openings.get("windows")
        parts.append(f"doors={len(doors) if isinstance(doors, list) else 0}")
        parts.append(f"windows={len(windows) if isinstance(windows, list) else 0}")
    camera = source_payload.get("camera")
    if isinstance(camera, Mapping):
        preset = camera.get("presetId") or camera.get("id") or camera.get("name")
        if isinstance(preset, str) and preset:
            parts.append(f"camera={preset}")
    return ", ".join(parts) if parts else "No room/opening metadata found."


def _bbox_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    min_x = _float_value(value.get("minX"))
    min_y = _float_value(value.get("minY"))
    width = _float_value(value.get("width"))
    height = _float_value(value.get("height"))
    if None in {min_x, min_y, width, height}:
        return ""
    return f", bbox_px=x:{min_x:.0f}, y:{min_y:.0f}, w:{width:.0f}, h:{height:.0f}"


def _api_key() -> str:
    for env_name in (_OPENAI_API_KEY_ENV, _OPENAI_IMAGE_API_KEY_ENV):
        value = os.getenv(env_name)
        if value:
            return value
    try:
        from config.openai_config import OpenAIConfig  # noqa: PLC0415

        return str(OpenAIConfig.OPENAI_API_KEY or "")
    except Exception:
        return ""


def _model() -> str:
    return os.getenv(_MODEL_ENV, _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _base_url() -> str:
    return (
        os.getenv(_BASE_URL_ENV)
        or os.getenv(_PREFLIGHT_BASE_URL_ENV)
        or _DEFAULT_BASE_URL
    )


def _json_mapping(value: object) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_safe(item) for key, item in value.items()}


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return cast(Sequence[object], value)
    return ()


def _bool_value(value: object, *, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return fallback


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string(value: object, *, fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback
