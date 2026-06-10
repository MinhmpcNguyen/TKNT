from __future__ import annotations

# ruff: noqa: E402

import argparse
import glob
import json
import logging
import mimetypes
import re
import sys
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.image_flow_eval.image_flow_eval.io import (
    load_manifest,
    write_json_file,
)  # noqa: E402 - CLI script bootstraps repo root for direct path execution.
from experiments.image_flow_eval.image_flow_eval.providers import (
    _best_approved_render_path,
)  # noqa: E402 - CLI script bootstraps repo root for direct path execution.
from experiments.image_flow_eval.image_flow_eval.types import JSONValue  # noqa: E402 - CLI script bootstraps repo root for direct path execution.

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST_GLOB = (
    "experiments/image_flow_eval/snapshot_audit/"
    "manifest.curated_core14_create.batch*.json"
)
_DEFAULT_CREATE_RUNS_ROOT = Path("experiments/image_flow_eval/runs/first_time")
_DEFAULT_OUTPUT_ROOT = Path(
    "experiments/image_flow_eval/runs/edit_from_first_time_manual"
)
_DEFAULT_MANIFEST_OUTPUT_PREFIX = Path("/private/tmp/tknt_edit_manual")
_DEFAULT_REPLACEMENT_IMAGE_DIR = Path("experiments/a_set_replace_image")
_DEFAULT_ANNOTATION_PATH = (
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "snapshot_audit"
    / "edit_bbox_annotations.json"
)
_DEFAULT_ADD_ANNOTATION_PATH = (
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "snapshot_audit"
    / "edit_add_bbox_annotations.json"
)
_DEFAULT_EDIT_SOURCE_APPROVAL_PATH = (
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "snapshot_audit"
    / "edit_source_approvals.json"
)
_APPROVED_SOURCE_VARIANT = "contract_objects_preflight_locked_geometry"
_DEFAULT_TARGET_COLOR = "muted forest green"
_DEFAULT_OPERATION_TYPE = "recolor"
_DEFAULT_OBJECT_ID = "manual_box_1"
_DEFAULT_OBJECT_NAME = "selected region"
_DEFAULT_PROMPT = (
    "Change only the selected object to muted forest green fabric or matte material; "
    "preserve its silhouette, position, scale, perspective, and every other object."
)
_DEFAULT_ADD_OBJECT_ID = "manual_add_box_1"
_DEFAULT_ADD_OBJECT_NAME = "wall art"
_DEFAULT_ADD_PROMPT = (
    "Add the wall art from the provided reference image inside the selected bbox only; "
    "keep it contained in that placement region and preserve camera, room shell, "
    "existing furniture, lighting, and everything outside the bbox exactly."
)
_EDIT_VARIANTS: list[JSONValue] = [
    {
        "id": "edit_prompt_image_only",
        "mode": "edit_prompt_image_only",
        "provider": "tknt_snapshot",
    },
    {
        "id": "edit_object_scope_preflight",
        "mode": "edit_object_scope_preflight",
        "provider": "tknt_snapshot",
    }
]
_EDIT_BOX_COLORS = (
    "#f97316",
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#0891b2",
)
_BATCH_PATTERN = re.compile(r"batch(\d+)")
_TARGET_PRIORITIES = {
    "sofa": 10,
    "sectional": 10,
    "armchair": 9,
    "lounge_chair": 9,
    "chair": 8,
    "wardrobe": 7,
    "cabinet": 7,
    "dresser": 7,
    "nightstand": 7,
    "side_table": 6,
    "coffee_table": 6,
    "desk": 6,
    "tv_console": 6,
    "bookshelf": 5,
    "console_table": 5,
    "bed": 4,
}
_EXCLUDED_TARGET_TOKENS = (
    "rug",
    "lamp",
    "light",
    "curtain",
    "blind",
    "window",
    "door",
    "pillow",
    "cushion",
    "blanket",
    "plant",
    "vase",
    "decor",
    "art",
)


@dataclass(frozen=True)
class ManualEditAnnotation:
    bbox_norm: dict[str, float]
    operation_type: str
    object_id: str
    object_name: str
    prompt: str
    target_color: str | None = None
    replacement_image_data_url: str | None = None

    def to_json(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "bbox_norm": dict(self.bbox_norm),
            "operation_type": self.operation_type,
            "object_id": self.object_id,
            "object_name": self.object_name,
            "prompt": self.prompt,
            "replacement_image_data_url": self.replacement_image_data_url,
            "target_color": self.target_color,
        }
        return payload


@dataclass(frozen=True)
class VisibleEditTarget:
    object_id: str
    object_name: str
    object_type: str
    bbox_norm: dict[str, float]
    area: float
    recommended: bool

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "object_type": self.object_type,
            "bbox_norm": dict(self.bbox_norm),
            "area": round(self.area, 4),
            "recommended": self.recommended,
        }


@dataclass(frozen=True)
class ManualEditCase:
    key: str
    batch_number: int
    case_id: str
    render_prompt: str | None
    snapshot_payload: Path
    snapshot_image: Path
    create_case_dir: Path
    edit_batch_output_dir: Path
    configured_render_path: Path
    approved_render_path: Path | None
    approved_variant: str
    approved_selection_mode: str
    approved_source_manual: bool
    guide_path: Path
    edit_targets: tuple[VisibleEditTarget, ...] = ()

    @property
    def edit_case_id(self) -> str:
        return f"{self.case_id}__edit_manual"

    def to_client_json(self) -> dict[str, JSONValue]:
        recommended_target = next(
            (target for target in self.edit_targets if target.recommended),
            self.edit_targets[0] if self.edit_targets else None,
        )
        return {
            "key": self.key,
            "batch_number": self.batch_number,
            "case_id": self.case_id,
            "edit_case_id": self.edit_case_id,
            "approved_variant": self.approved_variant,
            "approved_selection_mode": self.approved_selection_mode,
            "approved_source_manual": self.approved_source_manual,
            "approved_render_available": self.approved_render_path is not None,
            "approved_render_path": str(self.approved_render_path)
            if self.approved_render_path is not None
            else None,
            "configured_render_path": str(self.configured_render_path),
            "snapshot_payload": str(self.snapshot_payload),
            "snapshot_image": str(self.snapshot_image),
            "edit_targets": [target.to_json() for target in self.edit_targets],
            "recommended_edit_target": (
                recommended_target.to_json() if recommended_target is not None else None
            ),
            "render_url": f"/api/image?key={self.key}&kind=render",
            "snapshot_url": f"/api/image?key={self.key}&kind=snapshot",
            "guide_url": f"/api/image?key={self.key}&kind=guide",
        }


@dataclass(frozen=True)
class ManualEditBatch:
    batch_number: int
    source_manifest_path: Path
    defaults: Mapping[str, JSONValue]
    cases: tuple[ManualEditCase, ...]


@dataclass(frozen=True)
class ManualEditWriteResult:
    annotation_path: Path
    manifest_paths: tuple[Path, ...]
    run_command: str

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "annotation_path": str(self.annotation_path),
            "manifest_paths": [str(path) for path in self.manifest_paths],
            "run_command": self.run_command,
        }


class ManualEditBBoxApp:
    def __init__(
        self,
        *,
        batches: Sequence[ManualEditBatch],
        annotation_path: Path,
        manifest_output_prefix: Path,
        replacement_image_dir: Path,
        default_operation_type: str = _DEFAULT_OPERATION_TYPE,
        default_object_id: str = _DEFAULT_OBJECT_ID,
        default_object_name: str = _DEFAULT_OBJECT_NAME,
        default_prompt: str = _DEFAULT_PROMPT,
        default_target_color: str = _DEFAULT_TARGET_COLOR,
        include_guide_in_manifest: bool = False,
    ) -> None:
        self.batches = tuple(batches)
        self.annotation_path = annotation_path
        self.manifest_output_prefix = manifest_output_prefix
        self.replacement_image_dir = replacement_image_dir
        self.default_operation_type = default_operation_type
        self.default_object_id = default_object_id
        self.default_object_name = default_object_name
        self.default_prompt = default_prompt
        self.default_target_color = default_target_color
        self.include_guide_in_manifest = include_guide_in_manifest
        self.annotations = load_annotations(annotation_path)

    @property
    def cases_by_key(self) -> dict[str, ManualEditCase]:
        return {case.key: case for batch in self.batches for case in batch.cases}

    def state_json(self) -> dict[str, JSONValue]:
        return {
            "annotation_path": str(self.annotation_path),
            "manifest_output_prefix": str(self.manifest_output_prefix),
            "replacement_image_dir": str(self.replacement_image_dir),
            "replacement_images": {
                key: str(path)
                for key, path in sorted(
                    replacement_images_by_type(self.replacement_image_dir).items()
                )
            },
            "tool_defaults": {
                "operation_type": self.default_operation_type,
                "object_id": self.default_object_id,
                "object_name": self.default_object_name,
                "prompt": self.default_prompt,
                "target_color": self.default_target_color,
            },
            "include_guide_in_manifest": self.include_guide_in_manifest,
            "batches": [
                {
                    "batch_number": batch.batch_number,
                    "source_manifest_path": str(batch.source_manifest_path),
                    "cases": [case.to_client_json() for case in batch.cases],
                }
                for batch in self.batches
            ],
            "annotations": {
                key: annotation.to_json()
                for key, annotation in sorted(self.annotations.items())
            },
            "write_result": self.expected_write_result().to_json(),
        }

    def save_annotation(
        self,
        *,
        key: str,
        annotation: ManualEditAnnotation | None,
    ) -> ManualEditWriteResult:
        if key not in self.cases_by_key:
            raise ValueError(f"Unknown case key: {key}")
        if annotation is None:
            self.annotations.pop(key, None)
        else:
            self.annotations[key] = annotation
        return self.write_outputs(write_annotation_file=True)

    def write_outputs(self, *, write_annotation_file: bool) -> ManualEditWriteResult:
        if write_annotation_file:
            write_annotations(self.annotation_path, self.annotations)
        return build_manual_edit_manifests(
            batches=self.batches,
            annotations=self.annotations,
            annotation_path=self.annotation_path,
            manifest_output_prefix=self.manifest_output_prefix,
            replacement_image_dir=self.replacement_image_dir,
            include_guide_in_manifest=self.include_guide_in_manifest,
        )

    def expected_write_result(self) -> ManualEditWriteResult:
        manifest_paths = tuple(
            manifest_output_path_for_batch(
                self.manifest_output_prefix,
                batch.batch_number,
            )
            for batch in self.batches
        )
        return ManualEditWriteResult(
            annotation_path=self.annotation_path,
            manifest_paths=manifest_paths,
            run_command=_run_command_for_manifest_paths(manifest_paths),
        )


def bbox_norm_from_points(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> dict[str, float]:
    left = _clamp(min(start_x, end_x))
    top = _clamp(min(start_y, end_y))
    right = _clamp(max(start_x, end_x))
    bottom = _clamp(max(start_y, end_y))
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return {
        "x": round(left, 4),
        "y": round(top, 4),
        "width": round(width, 4),
        "height": round(height, 4),
    }


def load_manual_edit_batches(
    *,
    manifest_paths: Sequence[Path],
    create_runs_root: Path,
    output_root: Path,
    approval_path: Path | None = None,
) -> tuple[ManualEditBatch, ...]:
    approved_source_variants = (
        load_edit_source_approval_variants(approval_path)
        if approval_path is not None
        else {}
    )
    batches: list[ManualEditBatch] = []
    for index, manifest_path in enumerate(sorted(manifest_paths), start=1):
        batch_number = _batch_number_from_path(manifest_path, fallback=index)
        manifest = load_manifest(manifest_path)
        raw_manifest = _read_json_mapping(manifest_path)
        defaults = _json_mapping(raw_manifest.get("defaults", {}))
        create_batch_root = _resolve_create_batch_root(
            create_runs_root=create_runs_root,
            batch_number=batch_number,
        )
        edit_batch_output_dir = (
            output_root / f"curated_core14_create_batch{batch_number}"
        )
        cases: list[ManualEditCase] = []
        for case in manifest.cases:
            case_key = _case_key(batch_number=batch_number, case_id=case.id)
            manually_approved_variant = approved_source_variants.get(case_key)
            configured_render_path = (
                create_batch_root
                / case.id
                / (manually_approved_variant or _APPROVED_SOURCE_VARIANT)
                / "render.png"
            )
            if manually_approved_variant:
                approved_render_path = (
                    configured_render_path if configured_render_path.is_file() else None
                )
                approved_selection_mode = "exact"
            else:
                approved_render_path = _best_approved_render_path(
                    output_root=create_batch_root,
                    case_id=case.id,
                    configured_path=configured_render_path,
                )
                approved_selection_mode = "best_passed"
            approved_variant = (
                approved_render_path.parent.name
                if approved_render_path is not None
                else manually_approved_variant or _APPROVED_SOURCE_VARIANT
            )
            cases.append(
                ManualEditCase(
                    key=case_key,
                    batch_number=batch_number,
                    case_id=case.id,
                    render_prompt=case.render_prompt,
                    snapshot_payload=case.snapshot_payload or Path(),
                    snapshot_image=case.snapshot_image or Path(),
                    create_case_dir=create_batch_root / case.id,
                    edit_batch_output_dir=edit_batch_output_dir,
                    configured_render_path=configured_render_path,
                    approved_render_path=approved_render_path,
                    approved_variant=approved_variant,
                    approved_selection_mode=approved_selection_mode,
                    approved_source_manual=manually_approved_variant is not None,
                    guide_path=(
                        output_root
                        / "_bbox_guides"
                        / f"batch{batch_number}"
                        / f"{case.id}__manual_bbox.png"
                    ),
                    edit_targets=load_edit_targets(case.snapshot_payload or Path()),
                )
            )
        batches.append(
            ManualEditBatch(
                batch_number=batch_number,
                source_manifest_path=manifest_path,
                defaults=defaults,
                cases=tuple(cases),
            )
        )
    return tuple(batches)


def load_edit_targets(snapshot_payload_path: Path) -> tuple[VisibleEditTarget, ...]:
    if not snapshot_payload_path.is_file():
        return ()
    payload = _read_json_mapping(snapshot_payload_path)
    canvas_width, canvas_height = _canvas_size(payload)
    targets: list[VisibleEditTarget] = []
    for item in _visible_object_items(payload):
        object_id = _string(
            item.get("id") or item.get("objectId") or item.get("instance_id")
        )
        if not object_id:
            continue
        object_type = _string(item.get("type") or item.get("category") or object_id)
        object_name = _visible_object_name(item, fallback=object_type or object_id)
        bbox = _bbox_norm_from_screen_bbox(
            item.get("screenBboxPx")
            or item.get("screen_bbox_px")
            or item.get("bboxPx"),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        if bbox is None:
            continue
        area = bbox["width"] * bbox["height"]
        recommended = _is_recommended_target(
            object_id=object_id,
            object_type=object_type,
            object_name=object_name,
            bbox_norm=bbox,
        )
        targets.append(
            VisibleEditTarget(
                object_id=object_id,
                object_name=object_name,
                object_type=object_type or object_id,
                bbox_norm=bbox,
                area=area,
                recommended=recommended,
            )
        )
    sorted_targets = sorted(
        targets,
        key=lambda target: (
            not target.recommended,
            -_target_priority(target),
            -target.area,
            target.object_name,
        ),
    )
    recommended_object_id = next(
        (target.object_id for target in sorted_targets if target.recommended),
        None,
    )
    return tuple(
        VisibleEditTarget(
            object_id=target.object_id,
            object_name=target.object_name,
            object_type=target.object_type,
            bbox_norm=target.bbox_norm,
            area=target.area,
            recommended=target.object_id == recommended_object_id,
        )
        for target in sorted_targets
    )


def load_edit_source_approval_variants(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = _read_json_mapping(path)
    raw_cases = payload.get("cases", {})
    if not isinstance(raw_cases, dict):
        return {}
    approvals: dict[str, str] = {}
    for key, value in raw_cases.items():
        if not isinstance(key, str):
            continue
        variant = ""
        if isinstance(value, str):
            variant = value
        elif isinstance(value, dict):
            variant = _string(value.get("variant"))
        if variant:
            approvals[key] = variant
    return approvals


def load_annotations(path: Path) -> dict[str, ManualEditAnnotation]:
    if not path.is_file():
        return {}
    payload = _read_json_mapping(path)
    raw_cases = payload.get("cases", {})
    if not isinstance(raw_cases, dict):
        return {}
    annotations: dict[str, ManualEditAnnotation] = {}
    for key, value in raw_cases.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        try:
            annotations[key] = annotation_from_payload(value)
        except ValueError as exc:
            logger.warning("Skipping invalid bbox annotation %s: %s", key, exc)
    return annotations


def write_annotations(
    path: Path,
    annotations: Mapping[str, ManualEditAnnotation],
) -> None:
    payload: dict[str, JSONValue] = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "cases": {
            key: annotation.to_json() for key, annotation in sorted(annotations.items())
        },
    }
    write_json_file(path, payload)


def annotation_from_payload(payload: Mapping[str, object]) -> ManualEditAnnotation:
    bbox_raw = payload.get("bbox_norm")
    if not isinstance(bbox_raw, dict):
        raise ValueError("bbox_norm is required.")
    bbox_norm = _normalize_bbox_norm(cast(Mapping[str, object], bbox_raw))
    if bbox_norm is None:
        raise ValueError("bbox_norm is empty or too small.")
    operation_type = _string(payload.get("operation_type")) or _DEFAULT_OPERATION_TYPE
    if operation_type not in {"prompt", "add", "replace", "recolor", "remove"}:
        raise ValueError(f"Unsupported operation_type: {operation_type}")
    target_color = _string(payload.get("target_color")) or None
    if operation_type == "recolor" and target_color is None:
        target_color = _DEFAULT_TARGET_COLOR
    prompt = _string(payload.get("prompt")) or _default_prompt_for_operation(
        operation_type=operation_type,
        target_color=target_color,
    )
    object_id = _string(payload.get("object_id")) or _DEFAULT_OBJECT_ID
    object_name = _string(payload.get("object_name")) or _DEFAULT_OBJECT_NAME
    replacement_image_data_url = (
        _string(payload.get("replacement_image_data_url")) or None
    )
    return ManualEditAnnotation(
        bbox_norm=bbox_norm,
        operation_type=operation_type,
        object_id=object_id,
        object_name=object_name,
        prompt=prompt,
        target_color=target_color,
        replacement_image_data_url=replacement_image_data_url,
    )


def build_manual_edit_manifests(
    *,
    batches: Sequence[ManualEditBatch],
    annotations: Mapping[str, ManualEditAnnotation],
    annotation_path: Path,
    manifest_output_prefix: Path,
    replacement_image_dir: Path = (_REPO_ROOT / _DEFAULT_REPLACEMENT_IMAGE_DIR),
    include_guide_in_manifest: bool = False,
) -> ManualEditWriteResult:
    replacement_images = replacement_images_by_type(replacement_image_dir)
    manifest_paths: list[Path] = []
    for batch in batches:
        cases: list[JSONValue] = []
        for case in batch.cases:
            annotation = annotations.get(case.key)
            if annotation is None:
                continue
            if case.approved_render_path is None:
                logger.warning(
                    "Skipping %s because no approved render is available.",
                    case.key,
                )
                continue
            _ensure_create_case_link(case)
            create_edit_guide_image(
                source_image_path=case.approved_render_path,
                output_path=case.guide_path,
                annotation=annotation,
            )
            cases.append(
                _build_edit_case_entry(
                    case=case,
                    annotation=annotation,
                    replacement_images=replacement_images,
                    include_guide_in_manifest=include_guide_in_manifest,
                )
            )
        manifest_path = manifest_output_path_for_batch(
            manifest_output_prefix,
            batch.batch_number,
        )
        manifest_payload: dict[str, JSONValue] = {
            "defaults": dict(batch.defaults),
            "output_dir": str(
                batch.cases[0].edit_batch_output_dir
                if batch.cases
                else _REPO_ROOT / "experiments" / "image_flow_eval" / "runs"
            ),
            "cases": cases,
        }
        write_json_file(manifest_path, manifest_payload)
        manifest_paths.append(manifest_path)
    return ManualEditWriteResult(
        annotation_path=annotation_path,
        manifest_paths=tuple(manifest_paths),
        run_command=_run_command_for_manifest_paths(manifest_paths),
    )


def manifest_output_path_for_batch(prefix: Path, batch_number: int) -> Path:
    raw = str(prefix)
    if prefix.suffix == ".json":
        return prefix.with_name(f"{prefix.stem}_batch{batch_number}{prefix.suffix}")
    if raw.endswith("/") or prefix.is_dir():
        return prefix / f"tknt_edit_manual_batch{batch_number}.json"
    return prefix.with_name(f"{prefix.name}_batch{batch_number}.json")


def create_edit_guide_image(
    *,
    source_image_path: Path,
    output_path: Path,
    annotation: ManualEditAnnotation,
) -> Path:
    source = Image.open(source_image_path).convert("RGBA")
    width, height = source.size
    overlay = Image.new("RGBA", source.size, (15, 23, 42, 41))
    image = Image.alpha_composite(source, overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    color = _hex_to_rgb(_EDIT_BOX_COLORS[0])
    bbox = annotation.bbox_norm
    left = int(bbox["x"] * width)
    top = int(bbox["y"] * height)
    right = int((bbox["x"] + bbox["width"]) * width)
    bottom = int((bbox["y"] + bbox["height"]) * height)
    line_width = max(4, round(min(width, height) * 0.006))
    draw.rectangle(
        (left, top, right, bottom),
        fill=(*color, 36),
        outline=(*color, 255),
        width=line_width,
    )
    label = f"{annotation.object_id} {_operation_label(annotation.operation_type)}"
    font = ImageFont.load_default()
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 18
    label_height = label_box[3] - label_box[1] + 14
    label_left = int(_clamp_px(left + 6, 6, max(6, width - label_width - 6)))
    label_top = int(_clamp_px(top + 6, 6, max(6, height - label_height - 6)))
    label_rect = (
        label_left,
        label_top,
        label_left + label_width,
        label_top + label_height,
    )
    draw.rectangle(
        label_rect, fill=(255, 255, 255, 245), outline=(*color, 255), width=2
    )
    draw.text((label_left + 9, label_top + 7), label, fill=(17, 24, 39, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    manifest_paths = _resolve_manifest_paths(args.manifest_glob)
    if not manifest_paths:
        raise SystemExit(f"No manifests matched: {args.manifest_glob}")
    batches = load_manual_edit_batches(
        manifest_paths=manifest_paths,
        create_runs_root=_resolve_path(args.create_runs_root),
        output_root=_resolve_path(args.output_root),
        approval_path=_resolve_path(args.approval_path),
    )
    app = ManualEditBBoxApp(
        batches=batches,
        annotation_path=_resolve_path(args.annotation_path),
        manifest_output_prefix=Path(args.manifest_output_dir).expanduser(),
        replacement_image_dir=_resolve_path(args.replacement_image_dir),
        default_operation_type=args.default_operation_type,
        default_object_id=args.default_object_id,
        default_object_name=args.default_object_name,
        default_prompt=args.default_prompt,
        default_target_color=args.default_target_color,
        include_guide_in_manifest=args.include_guide_in_manifest,
    )
    server = _make_server(
        host=args.host,
        port=args.port,
        app=app,
    )
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    logger.info("Manual edit bbox tool: %s", url)
    logger.info("Press Ctrl+C to stop.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        server.server_close()
    return 0


def _build_edit_case_entry(
    *,
    case: ManualEditCase,
    annotation: ManualEditAnnotation,
    replacement_images: Mapping[str, Path],
    include_guide_in_manifest: bool,
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "id": case.edit_case_id,
        "snapshot_payload": str(case.snapshot_payload),
        "snapshot_image": str(case.snapshot_image),
        "render_prompt": case.render_prompt,
        "edit_prompt": annotation.prompt,
        "preset_selection": {},
        "edit_source_variant": {
            "case_id": case.case_id,
            "variant": case.approved_variant,
            "selection_mode": case.approved_selection_mode,
        },
        "edit_operations": [
            _edit_operation_payload(
                annotation,
                replacement_images=replacement_images,
            )
        ],
        "variants": list(_EDIT_VARIANTS),
    }
    if include_guide_in_manifest:
        payload["annotated_reference_image"] = str(case.guide_path)
    return payload


def _edit_operation_payload(
    annotation: ManualEditAnnotation,
    *,
    replacement_images: Mapping[str, Path],
) -> dict[str, JSONValue]:
    replacement_image = _replacement_image_for_annotation(
        annotation,
        replacement_images=replacement_images,
    )
    return {
        "object_id": annotation.object_id,
        "object_name": annotation.object_name,
        "operation_type": annotation.operation_type,
        "prompt": annotation.prompt,
        "bbox_norm": dict(annotation.bbox_norm),
        "replacement_image": str(replacement_image) if replacement_image else None,
        "target_color": annotation.target_color,
    }


def _ensure_create_case_link(case: ManualEditCase) -> None:
    if not case.create_case_dir.is_dir():
        logger.warning("Create case dir does not exist: %s", case.create_case_dir)
        return
    case.edit_batch_output_dir.mkdir(parents=True, exist_ok=True)
    link_path = case.edit_batch_output_dir / case.case_id
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == case.create_case_dir.resolve():
                return
        except OSError:
            pass
        raise FileExistsError(
            f"Cannot link create case into edit output; path already exists: {link_path}"
        )
    link_path.symlink_to(case.create_case_dir, target_is_directory=True)


def _run_command_for_manifest_paths(manifest_paths: Sequence[Path]) -> str:
    batch_numbers = [
        _batch_number_from_path(path, fallback=index)
        for index, path in enumerate(manifest_paths, start=1)
    ]
    batches = " ".join(str(number) for number in batch_numbers)
    if not manifest_paths:
        batches = "1 2 3 4 5"
    prefix = _manifest_prefix_for_command(manifest_paths)
    return (
        f"for B in {batches}; do\n"
        "  ./experiments/image_flow_eval/run_image_flow_eval.sh run \\\n"
        f"    --manifest {prefix}_batch${{B}}.json \\\n"
        "    --provider tknt_snapshot \\\n"
        "    --resume\n"
        "done"
    )


def _manifest_prefix_for_command(manifest_paths: Sequence[Path]) -> str:
    if not manifest_paths:
        return str(_DEFAULT_MANIFEST_OUTPUT_PREFIX)
    first = str(manifest_paths[0])
    return re.sub(r"_batch\d+\.json$", "", first)


def _make_server(
    *,
    host: str,
    port: int,
    app: ManualEditBBoxApp,
) -> ThreadingHTTPServer:
    class Handler(ManualEditBBoxRequestHandler):
        bbox_app = app

    return ThreadingHTTPServer((host, port), Handler)


class ManualEditBBoxRequestHandler(BaseHTTPRequestHandler):
    bbox_app: ManualEditBBoxApp

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(
                _HTML.encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return
        if parsed.path == "/api/state":
            self._send_json(self.bbox_app.state_json())
            return
        if parsed.path == "/api/image":
            self._handle_image_request(parsed.query)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/annotation":
                self._handle_save_annotation()
                return
            if parsed.path == "/api/write-manifests":
                result = self.bbox_app.write_outputs(write_annotation_file=True)
                self._send_json(
                    {
                        "ok": True,
                        "result": result.to_json(),
                        "state": self.bbox_app.state_json(),
                    }
                )
                return
        except (OSError, ValueError, FileExistsError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def log_message(self, format: str, *args: object) -> None:
        logger.debug(format, *args)

    def _handle_save_annotation(self) -> None:
        payload = self._read_json_body()
        key = _string(payload.get("key"))
        if not key:
            raise ValueError("key is required.")
        if payload.get("annotation") is None:
            result = self.bbox_app.save_annotation(key=key, annotation=None)
        else:
            annotation_raw = payload.get("annotation")
            if not isinstance(annotation_raw, dict):
                raise ValueError("annotation must be an object or null.")
            result = self.bbox_app.save_annotation(
                key=key,
                annotation=annotation_from_payload(annotation_raw),
            )
        self._send_json(
            {
                "ok": True,
                "result": result.to_json(),
                "state": self.bbox_app.state_json(),
            }
        )

    def _handle_image_request(self, query: str) -> None:
        params = parse_qs(query)
        key = params.get("key", [""])[0]
        kind = params.get("kind", ["render"])[0]
        case = self.bbox_app.cases_by_key.get(key)
        if case is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown case key.")
            return
        if kind == "render":
            path = case.approved_render_path
        elif kind == "snapshot":
            path = case.snapshot_image
        elif kind == "guide":
            path = case.guide_path
        else:
            path = None
        if path is None or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Image not found.")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), content_type=content_type)

    def _read_json_body(self) -> Mapping[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object.")
        return cast(Mapping[str, object], payload)

    def _send_json(self, payload: Mapping[str, object]) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"ok": False, "error": message}).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, payload: bytes, *, content_type: str) -> None:
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--manifest-glob", default=_DEFAULT_MANIFEST_GLOB)
    parser.add_argument("--create-runs-root", default=str(_DEFAULT_CREATE_RUNS_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--manifest-output-dir",
        default=str(_DEFAULT_MANIFEST_OUTPUT_PREFIX),
        help=(
            "Output prefix or directory. The planned value "
            "/private/tmp/tknt_edit_manual writes "
            "/private/tmp/tknt_edit_manual_batchN.json."
        ),
    )
    parser.add_argument(
        "--replacement-image-dir",
        default=str(_DEFAULT_REPLACEMENT_IMAGE_DIR),
        help=(
            "Directory of replacement reference images. Filenames should match "
            "object types, e.g. sofa.jpeg, wardrobe.jpg, bed.png."
        ),
    )
    parser.add_argument("--annotation-path", default=str(_DEFAULT_ANNOTATION_PATH))
    parser.add_argument(
        "--approval-path",
        default=str(_DEFAULT_EDIT_SOURCE_APPROVAL_PATH),
        help="Manual create-variant approvals written by select_edit_source_variants.py.",
    )
    parser.add_argument(
        "--default-operation-type",
        default=_DEFAULT_OPERATION_TYPE,
        choices=("prompt", "add", "replace", "recolor", "remove"),
        help="Initial operation selected in the UI for new annotations.",
    )
    parser.add_argument(
        "--default-object-id",
        default=_DEFAULT_OBJECT_ID,
        help="Initial object id for new manual annotations.",
    )
    parser.add_argument(
        "--default-object-name",
        default=_DEFAULT_OBJECT_NAME,
        help="Initial object name for new manual annotations.",
    )
    parser.add_argument(
        "--default-prompt",
        default=_DEFAULT_PROMPT,
        help="Initial prompt for new manual annotations.",
    )
    parser.add_argument(
        "--default-target-color",
        default=_DEFAULT_TARGET_COLOR,
        help="Initial target color for recolor annotations.",
    )
    parser.add_argument(
        "--include-guide-in-manifest",
        action="store_true",
        help="Pass the debug bbox guide PNG as annotated_reference_image.",
    )
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="Print the URL without opening the browser.",
    )
    parser.set_defaults(open_browser=True)
    return parser.parse_args(argv)


def _resolve_manifest_paths(pattern: str) -> list[Path]:
    raw = str(Path(pattern).expanduser())
    if Path(raw).is_absolute():
        return [Path(path).resolve() for path in sorted(glob.glob(raw))]
    return [path.resolve() for path in sorted(_REPO_ROOT.glob(raw))]


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (_REPO_ROOT / resolved).resolve()


def _batch_number_from_path(path: Path, *, fallback: int) -> int:
    match = _BATCH_PATTERN.search(path.name)
    if match:
        return int(match.group(1))
    return fallback


def _resolve_create_batch_root(
    *,
    create_runs_root: Path,
    batch_number: int,
) -> Path:
    if _BATCH_PATTERN.search(create_runs_root.name):
        return create_runs_root
    return create_runs_root / f"curated_core14_create_batch{batch_number}"


def _case_key(*, batch_number: int, case_id: str) -> str:
    return f"batch{batch_number}:{case_id}"


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _json_mapping(value: object) -> Mapping[str, JSONValue]:
    if not isinstance(value, dict):
        return {}
    return cast(Mapping[str, JSONValue], value)


def _canvas_size(payload: Mapping[str, object]) -> tuple[float, float]:
    canvas = payload.get("canvas")
    if isinstance(canvas, dict):
        width = _float_value(canvas.get("widthPx"), fallback=1536.0)
        height = _float_value(canvas.get("heightPx"), fallback=1024.0)
    else:
        width = _float_value(payload.get("widthPx"), fallback=1536.0)
        height = _float_value(payload.get("heightPx"), fallback=1024.0)
    return max(1.0, width), max(1.0, height)


def _visible_object_items(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = payload.get("visibleObjects") or payload.get("visible_objects") or []
    if isinstance(raw, dict):
        return [
            cast(Mapping[str, object], item)
            for item in raw.values()
            if isinstance(item, dict)
        ]
    if isinstance(raw, list):
        return [
            cast(Mapping[str, object], item) for item in raw if isinstance(item, dict)
        ]
    return []


def _visible_object_name(
    payload: Mapping[str, object],
    *,
    fallback: str,
) -> str:
    name = _string(payload.get("label") or payload.get("name") or payload.get("type"))
    return name.replace("_", " ") if name else fallback.replace("_", " ")


def _bbox_norm_from_screen_bbox(
    value: object,
    *,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    if {"minX", "minY", "maxX", "maxY"} <= value.keys():
        min_x = _float_value(value.get("minX"), fallback=0.0)
        min_y = _float_value(value.get("minY"), fallback=0.0)
        max_x = _float_value(value.get("maxX"), fallback=min_x)
        max_y = _float_value(value.get("maxY"), fallback=min_y)
    else:
        min_x = _float_value(value.get("minX") or value.get("x"), fallback=0.0)
        min_y = _float_value(value.get("minY") or value.get("y"), fallback=0.0)
        width = _float_value(value.get("width"), fallback=0.0)
        height = _float_value(value.get("height"), fallback=0.0)
        max_x = _float_value(value.get("maxX"), fallback=min_x + width)
        max_y = _float_value(value.get("maxY"), fallback=min_y + height)
    return _normalize_bbox_norm(
        {
            "x": min_x / canvas_width,
            "y": min_y / canvas_height,
            "width": max(0.0, max_x - min_x) / canvas_width,
            "height": max(0.0, max_y - min_y) / canvas_height,
        }
    )


def _is_recommended_target(
    *,
    object_id: str,
    object_type: str,
    object_name: str,
    bbox_norm: Mapping[str, float],
) -> bool:
    label = f"{object_id} {object_type} {object_name}".lower()
    if any(token in label for token in _EXCLUDED_TARGET_TOKENS):
        return False
    area = bbox_norm["width"] * bbox_norm["height"]
    if area < 0.01 or area > 0.28:
        return False
    if bbox_norm["width"] > 0.68 or bbox_norm["height"] > 0.68:
        return False
    return _target_priority_by_label(label) > 0


def _target_priority(target: VisibleEditTarget) -> int:
    return _target_priority_by_label(
        f"{target.object_id} {target.object_type} {target.object_name}".lower()
    )


def _target_priority_by_label(label: str) -> int:
    return max(
        (priority for token, priority in _TARGET_PRIORITIES.items() if token in label),
        default=0,
    )


def replacement_images_by_type(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image_type = _canonical_replacement_type(path.stem)
        images.setdefault(image_type, path.resolve())
    return images


def _replacement_image_for_annotation(
    annotation: ManualEditAnnotation,
    *,
    replacement_images: Mapping[str, Path],
) -> Path | None:
    if annotation.operation_type == "add":
        return replacement_images.get("wall_art")
    if annotation.operation_type != "replace":
        return None
    label = f"{annotation.object_id} {annotation.object_name}".lower()
    for token in _replacement_type_candidates(label):
        image = replacement_images.get(token)
        if image is not None:
            return image
    return None


def _replacement_type_candidates(label: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for token in re.split(r"[^a-z0-9]+", label.lower()):
        canonical = _canonical_replacement_type(token)
        if canonical:
            candidates.append(canonical)
    for key in _TARGET_PRIORITIES:
        if key in label:
            candidates.append(_canonical_replacement_type(key))
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _canonical_replacement_type(value: str) -> str:
    token = value.strip().lower().replace("-", "_")
    aliases = {
        "couch": "sofa",
        "sectional": "sofa",
        "closet": "wardrobe",
        "cabinet": "wardrobe",
        "dresser": "wardrobe",
        "armchair": "chair",
        "lounge_chair": "chair",
        "desk_table": "desk",
        "work_desk": "desk",
    }
    return aliases.get(token, token)


def _normalize_bbox_norm(value: Mapping[str, object]) -> dict[str, float] | None:
    try:
        x = float(value.get("x", 0.0))
        y = float(value.get("y", 0.0))
        width = float(value.get("width", 0.0))
        height = float(value.get("height", 0.0))
    except (TypeError, ValueError):
        return None
    bbox = bbox_norm_from_points(x, y, x + width, y + height)
    if bbox["width"] <= 0.002 or bbox["height"] <= 0.002:
        return None
    return bbox


def _default_prompt_for_operation(
    *,
    operation_type: str,
    target_color: str | None,
) -> str:
    if operation_type == "recolor":
        color = target_color or _DEFAULT_TARGET_COLOR
        return (
            f"Change only the selected object to {color} fabric or matte material; "
            "preserve its silhouette, position, scale, perspective, and every other object."
        )
    if operation_type == "remove":
        return (
            "Remove only the selected object or content inside the selected region; "
            "naturally inpaint the background and preserve every other object."
        )
    if operation_type == "add":
        return "Add the requested object inside the selected region only."
    if operation_type == "replace":
        return (
            "Replace only the selected region with the requested object or appearance."
        )
    return "Apply only the requested edit inside the selected region."


def _operation_label(operation_type: str) -> str:
    return operation_type.replace("_", " ").title()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_px(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _float_value(value: object, *, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TKNT Manual Edit BBox</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 14px 18px;
      background: rgba(255, 255, 255, 0.94);
      border-bottom: 1px solid #e2e8f0;
      backdrop-filter: blur(10px);
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }
    h1 { font-size: 18px; margin: 0; }
    button, select, input, textarea {
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 9px 10px;
      font: inherit;
      background: white;
    }
    button {
      cursor: pointer;
      background: #0f172a;
      color: white;
      border-color: #0f172a;
      font-weight: 700;
    }
    button.secondary { background: white; color: #0f172a; }
    button.danger { background: #dc2626; border-color: #dc2626; }
    main {
      display: grid;
      grid-template-columns: minmax(240px, 320px) 1fr minmax(280px, 360px);
      gap: 16px;
      padding: 16px;
    }
    aside, section, .panel {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      box-shadow: 0 12px 40px rgba(15, 23, 42, 0.06);
    }
    aside { padding: 10px; max-height: calc(100vh - 88px); overflow: auto; }
    .case-item {
      width: 100%;
      display: block;
      text-align: left;
      margin: 0 0 8px;
      padding: 10px;
      background: #f8fafc;
      color: #0f172a;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      font-weight: 600;
    }
    .case-item.active { outline: 3px solid #f97316; background: #fff7ed; }
    .case-item.saved::after { content: " saved"; color: #16a34a; font-weight: 800; }
    .case-item.missing { opacity: 0.55; }
    .case-target { display: block; margin-top: 4px; color: #64748b; font-size: 11px; font-weight: 700; }
    .viewer { padding: 12px; min-width: 0; }
    .image-wrap {
      position: relative;
      width: 100%;
      background: #020617;
      border-radius: 14px;
      overflow: hidden;
      user-select: none;
    }
    #renderImage {
      display: block;
      width: 100%;
      height: auto;
    }
    #overlayCanvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .side { padding: 14px; max-height: calc(100vh - 88px); overflow: auto; }
    .field { margin-bottom: 12px; }
    .field label { display: block; font-size: 12px; font-weight: 800; color: #475569; margin-bottom: 5px; }
    .field input, .field textarea, .field select { width: 100%; }
    .field textarea { min-height: 88px; resize: vertical; }
    .bbox-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .bbox-grid code {
      display: block;
      background: #f1f5f9;
      padding: 8px;
      border-radius: 10px;
      font-size: 12px;
    }
    .target-list {
      display: grid;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
      padding-right: 4px;
    }
    .target-item {
      width: 100%;
      text-align: left;
      display: block;
      color: #0f172a;
      background: #f8fafc;
      border-color: #e2e8f0;
      font-weight: 650;
    }
    .target-item.recommended {
      background: #fff7ed;
      border-color: #fed7aa;
      color: #9a3412;
    }
    .target-meta {
      display: block;
      margin-top: 3px;
      color: #64748b;
      font-size: 11px;
      font-weight: 600;
    }
    .status {
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 10px;
      font-size: 12px;
      color: #334155;
      max-height: 220px;
      overflow: auto;
    }
    .thumb { width: 100%; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 8px; }
    .muted { color: #64748b; font-size: 12px; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      aside, .side { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TKNT Manual Edit BBox</h1>
      <div class="muted">Draw on the clean approved render. The clean render remains the edit source; bbox_norm creates the edit mask.</div>
    </div>
    <div>
      <button class="secondary" id="writeBtn">Write manifests</button>
    </div>
  </header>
  <main>
    <aside>
      <div id="caseList"></div>
    </aside>
    <section class="viewer">
      <div class="image-wrap">
        <img id="renderImage" alt="approved render" />
        <canvas id="overlayCanvas"></canvas>
      </div>
      <p class="muted" id="caseMeta"></p>
    </section>
    <section class="side">
      <div class="field">
        <label>Edit Targets From Snapshot</label>
        <div class="target-list" id="targetList"></div>
        <div class="muted">Pick one object per case. Recommended is the top suggestion; other rows are fallback choices. Adjust the box manually if the snapshot bbox is too broad.</div>
      </div>
      <div class="field">
        <label>Operation</label>
        <select id="operationType">
          <option value="recolor">recolor</option>
          <option value="replace">replace</option>
          <option value="remove">remove</option>
          <option value="add">add</option>
          <option value="prompt">prompt</option>
        </select>
      </div>
      <div class="field">
        <label>Object ID</label>
        <input id="objectId" value="manual_box_1" />
      </div>
      <div class="field">
        <label>Object Name</label>
        <input id="objectName" value="selected region" />
      </div>
      <div class="field">
        <label>Target Color</label>
        <input id="targetColor" value="muted forest green" />
      </div>
      <div class="field">
        <label>Prompt</label>
        <textarea id="prompt">Change only the selected object to muted forest green fabric or matte material; preserve its silhouette, position, scale, perspective, and every other object.</textarea>
      </div>
      <div class="field">
        <label>BBox Norm</label>
        <div class="bbox-grid" id="bboxReadout"></div>
      </div>
      <div style="display:flex; gap:8px; margin-bottom:12px;">
        <button id="saveBtn">Save bbox</button>
        <button class="danger" id="clearBtn">Clear</button>
      </div>
      <div class="field">
        <label>Source 3D Snapshot</label>
        <img class="thumb" id="snapshotImage" alt="source 3D snapshot" />
      </div>
      <div class="field">
        <label>Status</label>
        <div class="status" id="statusBox"></div>
      </div>
    </section>
  </main>
  <script>
    const state = { data: null, cases: [], activeKey: null, bbox: null, drawing: null };
    const image = document.getElementById('renderImage');
    const snapshotImage = document.getElementById('snapshotImage');
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const caseList = document.getElementById('caseList');
    const statusBox = document.getElementById('statusBox');
    const caseMeta = document.getElementById('caseMeta');
    const bboxReadout = document.getElementById('bboxReadout');
    const targetList = document.getElementById('targetList');
    const fields = {
      operationType: document.getElementById('operationType'),
      objectId: document.getElementById('objectId'),
      objectName: document.getElementById('objectName'),
      targetColor: document.getElementById('targetColor'),
      prompt: document.getElementById('prompt'),
    };

    function clamp(value) { return Math.max(0, Math.min(1, value)); }
    function currentCase() { return state.cases.find((item) => item.key === state.activeKey) ?? null; }
    function currentAnnotation() { return state.data?.annotations?.[state.activeKey] ?? null; }
    function defaultPrompt(operationType, targetColor) {
      if (operationType === 'recolor') {
        return `Change only the selected object to ${targetColor || 'muted forest green'} fabric or matte material; preserve its silhouette, position, scale, perspective, and every other object.`;
      }
      if (operationType === 'remove') {
        return 'Remove only the selected object or content inside the selected region; naturally inpaint the background and preserve every other object.';
      }
      if (operationType === 'add') return 'Add the requested object inside the selected region only.';
      if (operationType === 'replace') return 'Replace only the selected region with the requested object or appearance.';
      return 'Apply only the requested edit inside the selected region.';
    }
    function setStatus(value) { statusBox.textContent = value; }
    function flattenCases(data) {
      return data.batches.flatMap((batch) => batch.cases.map((item) => ({ ...item, batch_number: batch.batch_number })));
    }
    async function loadState() {
      const response = await fetch('/api/state');
      state.data = await response.json();
      state.cases = flattenCases(state.data);
      if (!state.activeKey && state.cases.length > 0) state.activeKey = state.cases[0].key;
      renderCaseList();
      loadActiveCase();
      setStatus(state.data.write_result?.run_command ?? '');
    }
    function renderCaseList() {
      caseList.innerHTML = '';
      for (const item of state.cases) {
        const button = document.createElement('button');
        button.className = [
          'case-item',
          item.key === state.activeKey ? 'active' : '',
          state.data.annotations[item.key] ? 'saved' : '',
          item.approved_render_available ? '' : 'missing',
        ].join(' ');
        const recommended = item.recommended_edit_target;
        button.innerHTML = '';
        const title = document.createElement('span');
        title.textContent = `B${item.batch_number} · ${item.case_id}`;
        button.appendChild(title);
        if (recommended) {
          const target = document.createElement('span');
          target.className = 'case-target';
          target.textContent = `edit: ${recommended.object_name} · ${recommended.object_id}`;
          button.appendChild(target);
        }
        button.onclick = () => {
          state.activeKey = item.key;
          renderCaseList();
          loadActiveCase();
        };
        caseList.appendChild(button);
      }
    }
    function loadActiveCase() {
      const item = currentCase();
      if (!item) return;
      const annotation = currentAnnotation();
      const defaults = state.data?.tool_defaults ?? {};
      state.bbox = annotation?.bbox_norm ?? null;
      fields.operationType.value = annotation?.operation_type ?? defaults.operation_type ?? 'recolor';
      fields.objectId.value = annotation?.object_id ?? defaults.object_id ?? 'manual_box_1';
      fields.objectName.value = annotation?.object_name ?? defaults.object_name ?? 'selected region';
      fields.targetColor.value = annotation?.target_color ?? defaults.target_color ?? 'muted forest green';
      fields.prompt.value = annotation?.prompt ?? defaults.prompt ?? defaultPrompt(fields.operationType.value, fields.targetColor.value);
      image.src = item.render_url + '&t=' + Date.now();
      snapshotImage.src = item.snapshot_url + '&t=' + Date.now();
      const sourceMode = item.approved_source_manual ? 'manual exact' : item.approved_selection_mode;
      caseMeta.textContent = `${item.case_id} · approved variant: ${item.approved_variant} · source: ${sourceMode} · ${item.approved_render_path ?? 'missing render'}`;
      renderTargetList();
      updateReadout();
      resizeCanvas();
    }
    function renderTargetList() {
      const item = currentCase();
      targetList.innerHTML = '';
      const targets = item?.edit_targets ?? [];
      if (targets.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'muted';
        empty.textContent = 'No visible object bbox found in snapshot_payload.';
        targetList.appendChild(empty);
        return;
      }
      for (const target of targets) {
        const button = document.createElement('button');
        button.className = ['target-item', target.recommended ? 'recommended' : ''].join(' ');
        const title = document.createElement('span');
        title.textContent = `${target.recommended ? 'Recommended · ' : ''}${target.object_name} · ${target.object_id}`;
        const meta = document.createElement('span');
        meta.className = 'target-meta';
        meta.textContent = `type=${target.object_type} · area=${Number(target.area).toFixed(3)} · bbox=${formatBbox(target.bbox_norm)}`;
        button.appendChild(title);
        button.appendChild(meta);
        button.onclick = () => applyTarget(target);
        targetList.appendChild(button);
      }
    }
    function applyTarget(target) {
      state.bbox = { ...target.bbox_norm };
      const defaults = state.data?.tool_defaults ?? {};
      fields.objectId.value = target.object_id || 'manual_box_1';
      fields.objectName.value = target.object_name || target.object_type || 'selected region';
      if (!fields.targetColor.value) fields.targetColor.value = 'muted forest green';
      fields.prompt.value = defaults.prompt ?? defaultPrompt(fields.operationType.value, fields.targetColor.value).replace('selected object', fields.objectName.value || 'selected object');
      updateReadout();
      drawOverlay();
    }
    function formatBbox(box) {
      return `${Number(box.x).toFixed(3)},${Number(box.y).toFixed(3)},${Number(box.width).toFixed(3)},${Number(box.height).toFixed(3)}`;
    }
    function resizeCanvas() {
      const rect = image.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      drawOverlay();
    }
    function drawOverlay() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const box = state.drawing ?? state.bbox;
      if (!box) return;
      const x = box.x * rect.width;
      const y = box.y * rect.height;
      const w = box.width * rect.width;
      const h = box.height * rect.height;
      ctx.save();
      ctx.fillStyle = 'rgba(249, 115, 22, 0.14)';
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = Math.max(3, Math.min(rect.width, rect.height) * 0.006);
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
    function pointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: clamp((event.clientX - rect.left) / Math.max(1, rect.width)),
        y: clamp((event.clientY - rect.top) / Math.max(1, rect.height)),
      };
    }
    function bboxFromPoints(start, end) {
      const left = Math.min(start.x, end.x);
      const top = Math.min(start.y, end.y);
      const right = Math.max(start.x, end.x);
      const bottom = Math.max(start.y, end.y);
      return {
        x: Number(left.toFixed(4)),
        y: Number(top.toFixed(4)),
        width: Number((right - left).toFixed(4)),
        height: Number((bottom - top).toFixed(4)),
      };
    }
    function updateReadout() {
      const box = state.bbox;
      if (!box) {
        bboxReadout.innerHTML = '<code>Draw a box on the render.</code>';
        return;
      }
      bboxReadout.innerHTML = ['x', 'y', 'width', 'height']
        .map((key) => `<code>${key}: ${Number(box[key]).toFixed(4)}</code>`)
        .join('');
    }
    canvas.addEventListener('pointerdown', (event) => {
      const start = pointFromEvent(event);
      state.drawingStart = start;
      state.drawing = { x: start.x, y: start.y, width: 0, height: 0 };
      canvas.setPointerCapture(event.pointerId);
      drawOverlay();
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!state.drawingStart) return;
      state.drawing = bboxFromPoints(state.drawingStart, pointFromEvent(event));
      drawOverlay();
    });
    canvas.addEventListener('pointerup', () => {
      if (!state.drawingStart) return;
      state.bbox = state.drawing;
      state.drawing = null;
      state.drawingStart = null;
      updateReadout();
      drawOverlay();
    });
    image.addEventListener('load', resizeCanvas);
    window.addEventListener('resize', resizeCanvas);
    fields.operationType.addEventListener('change', () => {
      fields.prompt.value = defaultPrompt(fields.operationType.value, fields.targetColor.value);
    });
    fields.targetColor.addEventListener('input', () => {
      if (fields.operationType.value === 'recolor') {
        fields.prompt.value = defaultPrompt(fields.operationType.value, fields.targetColor.value);
      }
    });
    document.getElementById('saveBtn').onclick = async () => {
      if (!state.activeKey || !state.bbox) return;
      const annotation = {
        bbox_norm: state.bbox,
        operation_type: fields.operationType.value,
        object_id: fields.objectId.value || 'manual_box_1',
        object_name: fields.objectName.value || 'selected region',
        prompt: fields.prompt.value,
        target_color: fields.operationType.value === 'recolor' ? fields.targetColor.value : null,
        replacement_image_data_url: null,
      };
      const response = await fetch('/api/annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: state.activeKey, annotation }),
      });
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error);
      state.data = payload.state;
      state.cases = flattenCases(state.data);
      renderCaseList();
      setStatus(payload.result.run_command);
    };
    document.getElementById('clearBtn').onclick = async () => {
      if (!state.activeKey) return;
      const response = await fetch('/api/annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: state.activeKey, annotation: null }),
      });
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error);
      state.data = payload.state;
      state.cases = flattenCases(state.data);
      state.bbox = null;
      renderCaseList();
      updateReadout();
      drawOverlay();
      setStatus(payload.result.run_command);
    };
    document.getElementById('writeBtn').onclick = async () => {
      const response = await fetch('/api/write-manifests', { method: 'POST' });
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error);
      state.data = payload.state;
      state.cases = flattenCases(state.data);
      renderCaseList();
      setStatus(payload.result.run_command);
    };
    loadState().catch((error) => setStatus(String(error?.stack ?? error)));
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
