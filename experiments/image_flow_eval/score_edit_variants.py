from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import webbrowser
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUN_ROOTS = (
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "runs"
    / "edit_manual_by_operation_repeat1",
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "runs"
    / "edit_manual_by_operation_repeat2",
)
_DEFAULT_SCORE_PATH = (
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "snapshot_audit"
    / "edit_variant_human_scores.json"
)
_EDIT_VARIANTS = ("edit_prompt_image_only", "edit_object_scope_preflight")
_DISPLAY_METRICS = (
    "edit_instruction_fidelity",
    "edit_scope_fidelity",
    "layout_gate_score",
    "layout_dino_patch_similarity",
    "layout_edge_structure",
    "layout_depth_structure",
)
_SCHEMA_VERSION = 1
_CRITERIA_VERSION = "edit_manual_v1"


@dataclass(frozen=True)
class Criterion:
    id: str
    label: str
    help_text: str

    def to_json(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "help_text": self.help_text,
        }


_COMMON_CRITERIA = (
    Criterion(
        id="camera_room_preserved",
        label="Camera/room preserved",
        help_text="Camera, crop, walls, openings, floor, and global perspective match the original render.",
    ),
    Criterion(
        id="other_objects_preserved",
        label="Other objects preserved",
        help_text="Objects outside the requested edit region keep their shape, position, scale, and material identity.",
    ),
    Criterion(
        id="no_artifacts",
        label="No artifacts/guide marks",
        help_text="No bbox lines, labels, ghosting, smeared artifacts, or obvious inpainting marks remain.",
    ),
)

_CRITERIA_BY_OPERATION: dict[str, tuple[Criterion, ...]] = {
    "recolor": (
        Criterion(
            id="target_recolored",
            label="Target color/material changed",
            help_text="The intended target visibly follows the requested recolor/material.",
        ),
        Criterion(
            id="target_shape_preserved",
            label="Target geometry/details preserved",
            help_text="Target silhouette, footprint, handles, seams, panels, drawers, and occluders are preserved.",
        ),
        *_COMMON_CRITERIA,
    ),
    "replace": (
        Criterion(
            id="correct_replacement_identity",
            label="Correct replacement object",
            help_text="The replacement matches the intended/reference object category and identity.",
        ),
        Criterion(
            id="replacement_footprint_preserved",
            label="Footprint/scale/perspective preserved",
            help_text="Replacement stays inside the intended object footprint/bbox with plausible scale and perspective.",
        ),
        *_COMMON_CRITERIA,
    ),
    "remove": (
        Criterion(
            id="target_removed",
            label="Target removed",
            help_text="The selected object is actually removed, not merely restyled or partly left behind.",
        ),
        Criterion(
            id="background_repaired",
            label="Background repaired plausibly",
            help_text="The newly exposed wall/floor/furniture surface is plausible and consistent with the original render.",
        ),
        *_COMMON_CRITERIA,
    ),
    "add": (
        Criterion(
            id="object_added_in_bbox",
            label="Object added inside bbox",
            help_text="The new object appears in the requested placement box, not on another object or wall.",
        ),
        Criterion(
            id="added_object_identity_scale",
            label="Added object identity/scale correct",
            help_text="The added object follows the reference/request with plausible size, orientation, and perspective.",
        ),
        *_COMMON_CRITERIA,
    ),
}


@dataclass(frozen=True)
class MetricSummary:
    name: str
    status: str
    value: float | None

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
        }


@dataclass(frozen=True)
class EditOperationSummary:
    object_id: str
    object_name: str | None
    operation_type: str | None
    prompt: str | None
    target_color: str | None
    bbox_provided: bool
    replacement_image_provided: bool

    def to_json(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "operation_type": self.operation_type,
            "prompt": self.prompt,
            "target_color": self.target_color,
            "bbox_provided": self.bbox_provided,
            "replacement_image_provided": self.replacement_image_provided,
        }


@dataclass(frozen=True)
class EditVariantCandidate:
    variant: str
    dir_path: Path
    edit_image: Path
    original_render: Path | None
    source_3d: Path | None
    guide_image: Path | None
    metrics: tuple[MetricSummary, ...]
    edit_operations: tuple[EditOperationSummary, ...]
    provider_status: str | None
    runtime_s: float | None
    total_tokens: float | None

    def to_json(self, image_keys: Mapping[tuple[str, str, str], str], item_key: str) -> dict[str, object]:
        return {
            "variant": self.variant,
            "dir_path": str(self.dir_path),
            "edit_image": str(self.edit_image),
            "original_render": str(self.original_render) if self.original_render else None,
            "source_3d": str(self.source_3d) if self.source_3d else None,
            "guide_image": str(self.guide_image) if self.guide_image else None,
            "edit_url": f"/api/image?key={image_keys[(item_key, self.variant, 'edit')]}",
            "original_url": (
                f"/api/image?key={image_keys[(item_key, self.variant, 'original')]}"
                if self.original_render
                else None
            ),
            "source_url": (
                f"/api/image?key={image_keys[(item_key, self.variant, 'source')]}"
                if self.source_3d
                else None
            ),
            "guide_url": (
                f"/api/image?key={image_keys[(item_key, self.variant, 'guide')]}"
                if self.guide_image
                else None
            ),
            "metrics": [metric.to_json() for metric in self.metrics],
            "edit_operations": [operation.to_json() for operation in self.edit_operations],
            "provider_status": self.provider_status,
            "runtime_s": self.runtime_s,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class EditScoreItem:
    key: str
    repeat: str
    operation: str
    batch: str
    case_id: str
    variants: tuple[EditVariantCandidate, ...]

    def to_json(self, image_keys: Mapping[tuple[str, str, str], str]) -> dict[str, object]:
        criteria = _CRITERIA_BY_OPERATION.get(self.operation, _COMMON_CRITERIA)
        return {
            "key": self.key,
            "repeat": self.repeat,
            "operation": self.operation,
            "batch": self.batch,
            "case_id": self.case_id,
            "criteria": [criterion.to_json() for criterion in criteria],
            "variants": [
                variant.to_json(image_keys, self.key) for variant in self.variants
            ],
        }


class EditVariantScoringApp:
    def __init__(
        self,
        *,
        items: Sequence[EditScoreItem],
        score_path: Path,
    ) -> None:
        self.items = tuple(items)
        self.score_path = score_path
        self.image_paths: dict[str, Path] = {}
        self.image_keys: dict[tuple[str, str, str], str] = {}
        self.scores = _load_scores(score_path)
        self._register_images()

    @property
    def items_by_key(self) -> dict[str, EditScoreItem]:
        return {item.key: item for item in self.items}

    def state_json(self) -> dict[str, object]:
        return {
            "score_path": str(self.score_path),
            "summary_csv_path": str(_summary_csv_path(self.score_path)),
            "schema_version": _SCHEMA_VERSION,
            "criteria_version": _CRITERIA_VERSION,
            "items": [item.to_json(self.image_keys) for item in self.items],
            "scores": self.scores.get("scores", {}),
            "summary": self.summary_json(),
        }

    def save_score(
        self,
        *,
        item_key: str,
        variant: str,
        criteria: Mapping[str, object],
        notes: str | None,
    ) -> dict[str, object]:
        item = self.items_by_key.get(item_key)
        if item is None:
            raise ValueError(f"Unknown item key: {item_key}")
        if variant not in {candidate.variant for candidate in item.variants}:
            raise ValueError(f"Unknown variant for item {item_key}: {variant}")

        criteria_ids = [criterion.id for criterion in _CRITERIA_BY_OPERATION[item.operation]]
        normalized_criteria = {
            criterion_id: bool(criteria.get(criterion_id)) for criterion_id in criteria_ids
        }
        score = sum(1 for passed in normalized_criteria.values() if passed)
        max_score = len(criteria_ids)
        now = datetime.now(UTC).isoformat()

        scores = self.scores.setdefault("scores", {})
        if not isinstance(scores, dict):
            scores = {}
            self.scores["scores"] = scores
        item_scores = scores.setdefault(item_key, {})
        if not isinstance(item_scores, dict):
            item_scores = {}
            scores[item_key] = item_scores
        item_scores[variant] = {
            "repeat": item.repeat,
            "operation": item.operation,
            "batch": item.batch,
            "case_id": item.case_id,
            "variant": variant,
            "criteria_version": _CRITERIA_VERSION,
            "criteria": normalized_criteria,
            "score": score,
            "max_score": max_score,
            "score_ratio": round(score / max(1, max_score), 6),
            "notes": (notes or "").strip(),
            "scored_at": now,
        }
        self.scores["schema_version"] = _SCHEMA_VERSION
        self.scores["criteria_version"] = _CRITERIA_VERSION
        self.scores["updated_at"] = now
        _write_json(self.score_path, self.scores)
        _write_summary_csv(
            _summary_csv_path(self.score_path),
            items=self.items,
            scores=self.scores,
        )
        return {
            "saved": True,
            "score": item_scores[variant],
            "summary": self.summary_json(),
        }

    def summary_json(self) -> dict[str, object]:
        rows = _score_rows(self.items, self.scores)
        aggregate: dict[str, dict[str, object]] = {}
        for row in rows:
            variant = str(row["variant"])
            bucket = aggregate.setdefault(
                variant,
                {
                    "count": 0,
                    "score_sum": 0.0,
                    "ratio_sum": 0.0,
                    "by_operation": defaultdict(lambda: {"count": 0, "ratio_sum": 0.0}),
                },
            )
            bucket["count"] = int(bucket["count"]) + 1
            bucket["score_sum"] = float(bucket["score_sum"]) + float(row["score"])
            bucket["ratio_sum"] = float(bucket["ratio_sum"]) + float(row["score_ratio"])
            by_operation = bucket["by_operation"]
            operation_bucket = by_operation[str(row["operation"])]
            operation_bucket["count"] += 1
            operation_bucket["ratio_sum"] += float(row["score_ratio"])

        serializable: dict[str, object] = {}
        for variant, bucket in aggregate.items():
            count = int(bucket["count"])
            by_operation_payload = {}
            for operation, operation_bucket in bucket["by_operation"].items():
                operation_count = int(operation_bucket["count"])
                by_operation_payload[operation] = {
                    "count": operation_count,
                    "mean_score_ratio": round(
                        float(operation_bucket["ratio_sum"]) / max(1, operation_count),
                        4,
                    ),
                }
            serializable[variant] = {
                "count": count,
                "mean_score": round(float(bucket["score_sum"]) / max(1, count), 4),
                "mean_score_ratio": round(
                    float(bucket["ratio_sum"]) / max(1, count),
                    4,
                ),
                "by_operation": by_operation_payload,
            }
        return serializable

    def _register_images(self) -> None:
        for item in self.items:
            for candidate in item.variants:
                self._register_image(item.key, candidate.variant, "edit", candidate.edit_image)
                if candidate.original_render is not None:
                    self._register_image(
                        item.key,
                        candidate.variant,
                        "original",
                        candidate.original_render,
                    )
                if candidate.source_3d is not None:
                    self._register_image(item.key, candidate.variant, "source", candidate.source_3d)
                if candidate.guide_image is not None:
                    self._register_image(item.key, candidate.variant, "guide", candidate.guide_image)

    def _register_image(
        self,
        item_key: str,
        variant: str,
        kind: str,
        path: Path,
    ) -> None:
        image_key = f"img_{len(self.image_paths) + 1}"
        self.image_keys[(item_key, variant, kind)] = image_key
        self.image_paths[image_key] = path


def scan_edit_runs(
    run_roots: Sequence[Path],
    *,
    variants: Sequence[str],
) -> tuple[EditScoreItem, ...]:
    grouped: dict[tuple[str, str, str, str], list[EditVariantCandidate]] = defaultdict(list)
    for run_root in run_roots:
        root = run_root.expanduser().resolve()
        if not root.is_dir():
            logger.warning("Run root does not exist: %s", root)
            continue
        repeat = root.name.removeprefix("edit_manual_by_operation_")
        for metrics_path in sorted(root.glob("*/*/*/*/metrics.json")):
            variant_dir = metrics_path.parent
            variant = variant_dir.name
            if variant not in variants:
                continue
            edit_image = variant_dir / "edit.png"
            if not edit_image.is_file():
                continue
            operation, batch, edit_case_id = metrics_path.relative_to(root).parts[:3]
            case_id = edit_case_id.removesuffix("__edit_manual")
            result = _read_mapping(variant_dir / "tknt_snapshot_result.json")
            metrics = _metric_summaries(metrics_path)
            edit_operations = _edit_operation_summaries(result)
            original_render = _existing_path(result.get("approved_render_source")) or _existing_path(
                result.get("render_image")
            )
            source_3d = _existing_path(result.get("source_image")) or _existing_path(
                variant_dir / "source_3d.png"
            )
            guide_image = _find_guide_image(variant_dir, result)
            runtime_s, total_tokens, provider_status = _provider_summary(result, metrics)
            grouped[(repeat, operation, batch, case_id)].append(
                EditVariantCandidate(
                    variant=variant,
                    dir_path=variant_dir,
                    edit_image=edit_image,
                    original_render=original_render,
                    source_3d=source_3d,
                    guide_image=guide_image,
                    metrics=metrics,
                    edit_operations=edit_operations,
                    provider_status=provider_status,
                    runtime_s=runtime_s,
                    total_tokens=total_tokens,
                )
            )

    items: list[EditScoreItem] = []
    variant_rank = {variant: index for index, variant in enumerate(variants)}
    for (repeat, operation, batch, case_id), candidates in sorted(grouped.items()):
        ordered = tuple(sorted(candidates, key=lambda item: variant_rank.get(item.variant, 999)))
        key = f"{repeat}:{operation}:{batch}:{case_id}"
        items.append(
            EditScoreItem(
                key=key,
                repeat=repeat,
                operation=operation,
                batch=batch,
                case_id=case_id,
                variants=ordered,
            )
        )
    return tuple(items)


def _metric_summaries(metrics_path: Path) -> tuple[MetricSummary, ...]:
    payload = _read_json(metrics_path)
    if not isinstance(payload, list):
        return ()
    by_name = {
        str(metric.get("name")): metric
        for metric in payload
        if isinstance(metric, dict) and metric.get("name")
    }
    summaries: list[MetricSummary] = []
    for name in _DISPLAY_METRICS:
        metric = by_name.get(name)
        if not isinstance(metric, dict):
            continue
        summaries.append(
            MetricSummary(
                name=name,
                status=str(metric.get("status") or "unknown"),
                value=_float_or_none(metric.get("value")),
            )
        )
    return tuple(summaries)


def _edit_operation_summaries(
    result: Mapping[str, object],
) -> tuple[EditOperationSummary, ...]:
    request = _mapping(_mapping(result.get("edit_result")).get("request"))
    operations = request.get("edit_operations")
    if not isinstance(operations, list):
        return ()
    summaries: list[EditOperationSummary] = []
    for operation in operations:
        payload = _mapping(operation)
        summaries.append(
            EditOperationSummary(
                object_id=str(payload.get("object_id") or ""),
                object_name=_string_or_none(payload.get("object_name")),
                operation_type=_string_or_none(payload.get("operation_type")),
                prompt=_string_or_none(payload.get("prompt")),
                target_color=_string_or_none(payload.get("target_color")),
                bbox_provided=bool(payload.get("bbox_provided")),
                replacement_image_provided=bool(payload.get("replacement_image_provided")),
            )
        )
    return tuple(summaries)


def _provider_summary(
    result: Mapping[str, object],
    metrics: Sequence[MetricSummary],
) -> tuple[float | None, float | None, str | None]:
    runtime_s = _float_or_none(result.get("provider_runtime_s"))
    provider_status = _string_or_none(result.get("provider_status"))
    total_tokens = None
    for metric in metrics:
        if metric.name == "provider_runtime" and runtime_s is None:
            runtime_s = metric.value
        if metric.name == "provider_token_usage":
            total_tokens = metric.value
    edit_result = _mapping(result.get("edit_result"))
    usage = _mapping(edit_result.get("usage"))
    if total_tokens is None:
        total_tokens = _float_or_none(usage.get("total_tokens"))
    return runtime_s, total_tokens, provider_status


def _find_guide_image(
    variant_dir: Path,
    result: Mapping[str, object],
) -> Path | None:
    candidates: list[object] = [
        _mapping(result.get("object_scope_edit_mask")).get("image"),
        _mapping(result.get("edge_silhouette_guide")).get("guide_image"),
        variant_dir / "edit_box_guide.png",
        variant_dir.parent / "_bbox_guides" / f"{variant_dir.parent.name}.png",
    ]
    for candidate in candidates:
        path = _existing_path(candidate)
        if path is not None:
            return path
    return None


def _load_scores(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "schema_version": _SCHEMA_VERSION,
            "criteria_version": _CRITERIA_VERSION,
            "scores": {},
        }
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Score file must contain a JSON object: {path}")
    payload.setdefault("schema_version", _SCHEMA_VERSION)
    payload.setdefault("criteria_version", _CRITERIA_VERSION)
    payload.setdefault("scores", {})
    return payload


def _score_rows(
    items: Sequence[EditScoreItem],
    scores_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    scores = _mapping(scores_payload.get("scores"))
    rows: list[dict[str, object]] = []
    for item in items:
        item_scores = _mapping(scores.get(item.key))
        for candidate in item.variants:
            score = _mapping(item_scores.get(candidate.variant))
            if not score:
                continue
            rows.append(
                {
                    "repeat": item.repeat,
                    "operation": item.operation,
                    "batch": item.batch,
                    "case_id": item.case_id,
                    "variant": candidate.variant,
                    "score": int(score.get("score") or 0),
                    "max_score": int(score.get("max_score") or 0),
                    "score_ratio": float(score.get("score_ratio") or 0.0),
                    "notes": str(score.get("notes") or ""),
                    "scored_at": str(score.get("scored_at") or ""),
                }
            )
    return rows


def _write_summary_csv(
    path: Path,
    *,
    items: Sequence[EditScoreItem],
    scores: Mapping[str, object],
) -> None:
    rows = _score_rows(items, scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "repeat",
                "operation",
                "batch",
                "case_id",
                "variant",
                "score",
                "max_score",
                "score_ratio",
                "notes",
                "scored_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _summary_csv_path(score_path: Path) -> Path:
    return score_path.with_suffix(".summary.csv")


def build_server(
    *,
    app: EditVariantScoringApp,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    class Handler(EditVariantScoreRequestHandler):
        scoring_app = app

    return ThreadingHTTPServer((host, port), Handler)


class EditVariantScoreRequestHandler(BaseHTTPRequestHandler):
    scoring_app: EditVariantScoringApp

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(_HTML)
            elif parsed.path == "/api/state":
                self._send_json(self.scoring_app.state_json())
            elif parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                key = (query.get("key") or [""])[0]
                self._send_image(key)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except (OSError, ValueError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/score":
                payload = self._read_json_body()
                response = self.scoring_app.save_score(
                    item_key=str(payload.get("item_key") or ""),
                    variant=str(payload.get("variant") or ""),
                    criteria=_mapping(payload.get("criteria")),
                    notes=_string_or_none(payload.get("notes")),
                )
                self._send_json(response)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        logger.info("web: " + format, *args)

    def _read_json_body(self) -> Mapping[str, object]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw or "{}")
        return _mapping(payload)

    def _send_json(self, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, key: str) -> None:
        path = self.scoring_app.image_paths.get(key)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    payload = _read_json(path)
    return _mapping(payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _existing_path(value: object) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        return None
    path = path.expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path if path.is_file() else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a local UI to manually score edit variants by checklist."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        dest="run_roots",
        help="Edit run root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--score-path",
        type=Path,
        default=_DEFAULT_SCORE_PATH,
        help="Where to save checklist scores JSON.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--list", action="store_true", help="List scanned items and exit.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    run_roots = tuple(args.run_roots or _DEFAULT_RUN_ROOTS)
    items = scan_edit_runs(run_roots, variants=_EDIT_VARIANTS)
    if args.list:
        for item in items:
            variants = ", ".join(candidate.variant for candidate in item.variants)
            print(f"{item.key}: {variants}")
        print(f"items={len(items)} variants={sum(len(item.variants) for item in items)}")
        return 0
    app = EditVariantScoringApp(items=items, score_path=args.score_path)
    server = build_server(app=app, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    logger.info("Loaded %s case/repeat items.", len(items))
    logger.info("Saving scores to %s", args.score_path)
    logger.info("Serving %s", url)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        server.server_close()
    return 0


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TKNT Edit Variant Manual Score</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f4f7fb; color: #111827; }
    header { position: sticky; top: 0; z-index: 4; background: rgba(255,255,255,0.96); border-bottom: 1px solid #dbe3ef; padding: 14px 18px; display: flex; gap: 16px; align-items: center; }
    h1 { font-size: 20px; margin: 0; }
    button, select { border: 1px solid #cbd5e1; background: white; border-radius: 10px; padding: 8px 10px; font-weight: 700; color: #111827; }
    button { cursor: pointer; }
    button.primary { background: #111827; color: white; border-color: #111827; }
    .layout { display: grid; grid-template-columns: 330px 1fr; min-height: calc(100vh - 58px); }
    .sidebar { border-right: 1px solid #dbe3ef; background: white; padding: 12px; overflow: auto; height: calc(100vh - 58px); box-sizing: border-box; }
    .content { padding: 16px; overflow: auto; }
    .filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
    .case-list { display: flex; flex-direction: column; gap: 8px; }
    .case-card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px; background: #f8fafc; cursor: pointer; }
    .case-card.active { border-color: #f97316; box-shadow: 0 0 0 2px rgba(249,115,22,0.18); background: #fff7ed; }
    .case-card.done { background: #ecfdf5; border-color: #86efac; }
    .muted { color: #64748b; font-size: 13px; }
    .topline { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 10px; }
    .title { font-size: 18px; font-weight: 850; }
    .images { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .panel { background: white; border: 1px solid #dbe3ef; border-radius: 16px; overflow: hidden; }
    .panel h2 { font-size: 16px; margin: 0; padding: 12px 14px; border-bottom: 1px solid #e2e8f0; }
    .image-wrap { padding: 12px; }
    img { max-width: 100%; border-radius: 10px; background: #e5e7eb; display: block; }
    .variant-grid { display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 14px; margin-top: 14px; }
    .variant-card { background: white; border: 1px solid #dbe3ef; border-radius: 16px; overflow: hidden; }
    .variant-head { padding: 12px 14px; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    .variant-name { font-size: 17px; font-weight: 850; }
    .score-pill { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #eff6ff; color: #1d4ed8; border-radius: 999px; padding: 5px 8px; font-size: 13px; }
    .checklist { padding: 12px 14px; display: grid; gap: 10px; }
    label.check { display: grid; grid-template-columns: 22px 1fr; gap: 8px; align-items: start; }
    label.check input { width: 18px; height: 18px; margin-top: 2px; }
    .help { color: #64748b; font-size: 12px; margin-top: 2px; line-height: 1.35; }
    textarea { width: 100%; min-height: 66px; resize: vertical; border: 1px solid #cbd5e1; border-radius: 10px; padding: 8px; box-sizing: border-box; font: inherit; }
    .metrics { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 12px; }
    .metric { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background: #f1f5f9; border-radius: 999px; padding: 4px 7px; }
    .operation { text-transform: uppercase; color: #ea580c; font-weight: 900; }
    .ops { padding: 0 14px 12px; color: #475569; font-size: 13px; }
    .summary { font-size: 13px; display: flex; gap: 12px; flex-wrap: wrap; }
    .summary span { background: #eef2ff; border-radius: 999px; padding: 5px 8px; }
  </style>
</head>
<body>
  <header>
    <h1>TKNT Edit Variant Manual Score</h1>
    <button id="prev">Prev</button>
    <button id="next" class="primary">Next</button>
    <div class="summary" id="summary"></div>
    <div class="muted" id="savePath"></div>
  </header>
  <main class="layout">
    <aside class="sidebar">
      <div class="filters">
        <select id="operationFilter"></select>
        <select id="repeatFilter"></select>
      </div>
      <div class="case-list" id="caseList"></div>
    </aside>
    <section class="content">
      <div class="topline">
        <div>
          <div class="title" id="caseTitle"></div>
          <div class="muted" id="caseMeta"></div>
        </div>
      </div>
      <div class="images">
        <div class="panel">
          <h2>Original approved render (baseline)</h2>
          <div class="image-wrap"><img id="originalImage" /></div>
        </div>
        <div class="panel">
          <h2>Source 3D / guide</h2>
          <div class="image-wrap"><img id="sourceImage" /></div>
        </div>
      </div>
      <div class="variant-grid" id="variantGrid"></div>
    </section>
  </main>
  <script>
    let state = null;
    let filteredItems = [];
    let activeIndex = 0;

    async function loadState() {
      const response = await fetch('/api/state');
      state = await response.json();
      document.getElementById('savePath').textContent = `scores: ${state.score_path}`;
      initFilters();
      applyFilters();
      renderSummary();
    }

    function initFilters() {
      const operations = ['all', ...new Set(state.items.map(item => item.operation))].sort();
      const repeats = ['all', ...new Set(state.items.map(item => item.repeat))].sort();
      fillSelect('operationFilter', operations);
      fillSelect('repeatFilter', repeats);
      document.getElementById('operationFilter').addEventListener('change', applyFilters);
      document.getElementById('repeatFilter').addEventListener('change', applyFilters);
      document.getElementById('prev').addEventListener('click', () => move(-1));
      document.getElementById('next').addEventListener('click', () => move(1));
    }

    function fillSelect(id, values) {
      const select = document.getElementById(id);
      select.innerHTML = '';
      values.forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
    }

    function applyFilters() {
      const op = document.getElementById('operationFilter').value;
      const repeat = document.getElementById('repeatFilter').value;
      filteredItems = state.items.filter(item =>
        (op === 'all' || item.operation === op) &&
        (repeat === 'all' || item.repeat === repeat)
      );
      activeIndex = Math.min(activeIndex, Math.max(0, filteredItems.length - 1));
      renderCaseList();
      renderActiveItem();
    }

    function move(delta) {
      if (!filteredItems.length) return;
      activeIndex = Math.max(0, Math.min(filteredItems.length - 1, activeIndex + delta));
      renderCaseList();
      renderActiveItem();
    }

    function renderCaseList() {
      const list = document.getElementById('caseList');
      list.innerHTML = '';
      filteredItems.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'case-card';
        if (index === activeIndex) card.classList.add('active');
        if (isItemDone(item)) card.classList.add('done');
        card.innerHTML = `
          <div><b>${item.batch} · ${item.repeat}</b> <span class="operation">${item.operation}</span></div>
          <div>${item.case_id}</div>
          <div class="muted">${item.variants.map(v => scoreText(item.key, v.variant)).join(' · ')}</div>
        `;
        card.addEventListener('click', () => {
          activeIndex = index;
          renderCaseList();
          renderActiveItem();
        });
        list.appendChild(card);
      });
    }

    function isItemDone(item) {
      const scores = state.scores[item.key] || {};
      return item.variants.every(variant => scores[variant.variant]);
    }

    function renderActiveItem() {
      const item = filteredItems[activeIndex];
      if (!item) return;
      document.getElementById('caseTitle').textContent = item.case_id;
      document.getElementById('caseMeta').textContent = `${item.repeat} · ${item.operation} · ${item.batch}`;
      const first = item.variants[0];
      document.getElementById('originalImage').src = first.original_url || first.edit_url;
      document.getElementById('sourceImage').src = first.source_url || first.guide_url || first.edit_url;

      const grid = document.getElementById('variantGrid');
      grid.innerHTML = '';
      item.variants.forEach(variant => {
        grid.appendChild(renderVariantCard(item, variant));
      });
    }

    function renderVariantCard(item, variant) {
      const saved = (state.scores[item.key] || {})[variant.variant] || {};
      const criteria = saved.criteria || {};
      const card = document.createElement('div');
      card.className = 'variant-card';
      card.innerHTML = `
        <div class="variant-head">
          <div class="variant-name">${variant.variant}</div>
          <div class="score-pill" data-score>${scoreText(item.key, variant.variant)}</div>
        </div>
        <div class="image-wrap"><img src="${variant.edit_url}" /></div>
        <div class="ops">${operationText(variant)}</div>
        <div class="metrics">${variant.metrics.map(metricBadge).join('')}</div>
      `;
      const checklist = document.createElement('div');
      checklist.className = 'checklist';
      item.criteria.forEach(criterion => {
        const label = document.createElement('label');
        label.className = 'check';
        const checked = criteria[criterion.id] ? 'checked' : '';
        label.innerHTML = `
          <input type="checkbox" data-criterion="${criterion.id}" ${checked} />
          <span><b>${criterion.label}</b><div class="help">${criterion.help_text}</div></span>
        `;
        checklist.appendChild(label);
      });
      const textarea = document.createElement('textarea');
      textarea.placeholder = 'Notes: what is wrong, why this variant wins/loses...';
      textarea.value = saved.notes || '';
      checklist.appendChild(textarea);
      checklist.addEventListener('change', () => saveCard(item, variant, card, textarea));
      textarea.addEventListener('blur', () => saveCard(item, variant, card, textarea));
      card.appendChild(checklist);
      return card;
    }

    async function saveCard(item, variant, card, textarea) {
      const criteria = {};
      card.querySelectorAll('input[type="checkbox"]').forEach(input => {
        criteria[input.dataset.criterion] = input.checked;
      });
      const response = await fetch('/api/score', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          item_key: item.key,
          variant: variant.variant,
          criteria,
          notes: textarea.value,
        }),
      });
      const payload = await response.json();
      if (!payload.saved) return;
      state.scores[item.key] = state.scores[item.key] || {};
      state.scores[item.key][variant.variant] = payload.score;
      state.summary = payload.summary;
      card.querySelector('[data-score]').textContent = scoreText(item.key, variant.variant);
      renderCaseList();
      renderSummary();
    }

    function scoreText(itemKey, variant) {
      const score = (state.scores[itemKey] || {})[variant];
      if (!score) return `${variant}: unscored`;
      return `${score.score}/${score.max_score}`;
    }

    function operationText(variant) {
      if (!variant.edit_operations.length) return 'No edit operation metadata.';
      return variant.edit_operations.map(op =>
        `<b>${op.operation_type || 'edit'}</b> · ${op.object_name || op.object_id} · ${op.prompt || ''}`
      ).join('<br/>');
    }

    function metricBadge(metric) {
      const value = metric.value == null ? '-' : Number(metric.value).toFixed(3);
      return `<span class="metric">${metric.name}=${value} ${metric.status}</span>`;
    }

    function renderSummary() {
      const target = document.getElementById('summary');
      const summary = state.summary || {};
      target.innerHTML = Object.entries(summary).map(([variant, payload]) =>
        `<span>${variant}: n=${payload.count}, mean=${Number(payload.mean_score_ratio).toFixed(3)}</span>`
      ).join('');
    }

    loadState();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
