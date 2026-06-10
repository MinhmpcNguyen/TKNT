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
_DEFAULT_APPROVAL_PATH = (
    _REPO_ROOT
    / "experiments"
    / "image_flow_eval"
    / "snapshot_audit"
    / "edit_source_approvals.json"
)
_BATCH_PATTERN = re.compile(r"batch(\d+)")
_DEFAULT_CONFIGURED_VARIANT = "contract_objects_preflight_locked_geometry"
_DEFAULT_SOURCE_VARIANTS = (
    "contract_objects_preflight_locked_geometry",
    "llm_visual_preflight",
    "contract_objects_preflight",
    "contract_objects",
    "simple_prompt",
    "single_edit_edge_silhouette_guided",
)
_SCORE_METRICS = ("structural_layout_score", "layout_gate_score", "layout_fidelity")
_DISPLAY_METRICS = (
    "structural_layout_score",
    "layout_gate_score",
    "layout_fidelity",
    "layout_edge_structure",
    "layout_depth_structure",
    "layout_dino_patch_similarity",
    "layout_dino_object_similarity",
    "detector_layout",
    "detector_spatial_relations",
    "preset_fidelity",
    "blank",
)


@dataclass(frozen=True)
class MetricSummary:
    name: str
    status: str
    value: float | None
    threshold: float | None

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class EditSourceCandidate:
    variant: str
    render_path: Path
    metrics_path: Path
    source_image_path: Path | None
    score_metric: str | None
    score: float | None
    score_status: str | None
    display_metrics: tuple[MetricSummary, ...]
    hard_check_failures: tuple[str, ...]
    reason: str | None

    @property
    def passed(self) -> bool:
        return self.score_status == "passed"

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "variant": self.variant,
            "render_path": str(self.render_path),
            "metrics_path": str(self.metrics_path),
            "source_image_path": str(self.source_image_path)
            if self.source_image_path is not None
            else None,
            "score_metric": self.score_metric,
            "score": self.score,
            "score_status": self.score_status,
            "passed": self.passed,
            "metrics": [metric.to_json() for metric in self.display_metrics],
            "hard_check_failures": list(self.hard_check_failures),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EditSourceApproval:
    batch_number: int
    case_id: str
    variant: str
    render_path: Path
    score_metric: str | None
    score: float | None
    selected_at: str

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "batch_number": self.batch_number,
            "case_id": self.case_id,
            "variant": self.variant,
            "render_path": str(self.render_path),
            "score_metric": self.score_metric,
            "score": self.score,
            "selected_at": self.selected_at,
        }


@dataclass(frozen=True)
class EditSourceSelectionCase:
    key: str
    batch_number: int
    case_id: str
    snapshot_image: Path | None
    source_image: Path | None
    create_case_dir: Path
    auto_variant: str | None
    candidates: tuple[EditSourceCandidate, ...]

    def to_client_json(
        self,
        approval: EditSourceApproval | None,
    ) -> dict[str, JSONValue]:
        selected_variant = (
            approval.variant if approval is not None else self.auto_variant
        )
        return {
            "key": self.key,
            "batch_number": self.batch_number,
            "case_id": self.case_id,
            "snapshot_image": str(self.snapshot_image)
            if self.snapshot_image is not None
            else None,
            "source_image": str(self.source_image) if self.source_image else None,
            "create_case_dir": str(self.create_case_dir),
            "auto_variant": self.auto_variant,
            "selected_variant": selected_variant,
            "selection_manual": approval is not None,
            "source_url": f"/api/image?key={self.key}&kind=source",
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


class EditSourceSelectorApp:
    def __init__(
        self,
        *,
        cases: Sequence[EditSourceSelectionCase],
        approval_path: Path,
    ) -> None:
        self.cases = tuple(cases)
        self.approval_path = approval_path
        self.approvals = load_edit_source_approvals(approval_path)

    @property
    def cases_by_key(self) -> dict[str, EditSourceSelectionCase]:
        return {case.key: case for case in self.cases}

    def state_json(self) -> dict[str, JSONValue]:
        return {
            "approval_path": str(self.approval_path),
            "cases": [
                case.to_client_json(self.approvals.get(case.key)) for case in self.cases
            ],
            "approvals": {
                key: approval.to_json()
                for key, approval in sorted(self.approvals.items())
            },
        }

    def save_approval(
        self,
        *,
        key: str,
        variant: str | None,
    ) -> None:
        case = self.cases_by_key.get(key)
        if case is None:
            raise ValueError(f"Unknown case key: {key}")
        if not variant:
            self.approvals.pop(key, None)
            write_edit_source_approvals(self.approval_path, self.approvals)
            return
        candidate = next(
            (item for item in case.candidates if item.variant == variant),
            None,
        )
        if candidate is None:
            raise ValueError(f"Unknown variant for {key}: {variant}")
        self.approvals[key] = EditSourceApproval(
            batch_number=case.batch_number,
            case_id=case.case_id,
            variant=candidate.variant,
            render_path=candidate.render_path,
            score_metric=candidate.score_metric,
            score=candidate.score,
            selected_at=datetime.now(UTC).isoformat(),
        )
        write_edit_source_approvals(self.approval_path, self.approvals)


def load_selection_cases(
    *,
    manifest_paths: Sequence[Path],
    create_runs_root: Path,
    variant_ids: Sequence[str] = _DEFAULT_SOURCE_VARIANTS,
) -> tuple[EditSourceSelectionCase, ...]:
    cases: list[EditSourceSelectionCase] = []
    allowed_variants = tuple(dict.fromkeys(variant_ids))
    for index, manifest_path in enumerate(sorted(manifest_paths), start=1):
        batch_number = _batch_number_from_path(manifest_path, fallback=index)
        manifest = load_manifest(manifest_path)
        create_batch_root = _resolve_create_batch_root(
            create_runs_root=create_runs_root,
            batch_number=batch_number,
        )
        for case in manifest.cases:
            create_case_dir = create_batch_root / case.id
            candidates = load_variant_candidates(
                create_case_dir,
                variant_ids=allowed_variants,
            )
            source_image = _source_image_for_case(
                snapshot_image=case.snapshot_image,
                candidates=candidates,
            )
            auto_variant = _auto_variant(
                create_batch_root=create_batch_root,
                case_id=case.id,
                candidates=candidates,
            )
            cases.append(
                EditSourceSelectionCase(
                    key=_case_key(batch_number=batch_number, case_id=case.id),
                    batch_number=batch_number,
                    case_id=case.id,
                    snapshot_image=case.snapshot_image,
                    source_image=source_image,
                    create_case_dir=create_case_dir,
                    auto_variant=auto_variant,
                    candidates=candidates,
                )
            )
    return tuple(cases)


def load_variant_candidates(
    create_case_dir: Path,
    *,
    variant_ids: Sequence[str] = _DEFAULT_SOURCE_VARIANTS,
) -> tuple[EditSourceCandidate, ...]:
    return _load_variant_candidates(
        create_case_dir,
        variant_ids=variant_ids,
    )


def _load_variant_candidates(
    create_case_dir: Path,
    *,
    variant_ids: Sequence[str],
) -> tuple[EditSourceCandidate, ...]:
    if not create_case_dir.is_dir():
        return ()
    allowed_variants = set(variant_ids)
    candidates: list[EditSourceCandidate] = []
    for variant_dir in sorted(create_case_dir.iterdir(), key=lambda path: path.name):
        if not variant_dir.is_dir():
            continue
        if variant_dir.name not in allowed_variants:
            continue
        render_path = variant_dir / "render.png"
        if not _is_file(render_path):
            continue
        metrics_path = variant_dir / "metrics.json"
        metric_items = _read_metric_items(metrics_path)
        score_metric, score, score_status = _score_from_metrics(metric_items)
        candidates.append(
            EditSourceCandidate(
                variant=variant_dir.name,
                render_path=render_path,
                metrics_path=metrics_path,
                source_image_path=_existing_path(variant_dir / "source_3d.png"),
                score_metric=score_metric,
                score=score,
                score_status=score_status,
                display_metrics=_display_metric_summaries(metric_items),
                hard_check_failures=_hard_check_failures(metric_items),
                reason=_layout_reason(metric_items),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.score is None,
                -(candidate.score or float("-inf")),
                candidate.variant,
            ),
        )
    )


def load_edit_source_approvals(path: Path) -> dict[str, EditSourceApproval]:
    if not path.is_file():
        return {}
    payload = _read_json_mapping(path)
    raw_cases = payload.get("cases", {})
    if not isinstance(raw_cases, dict):
        return {}
    approvals: dict[str, EditSourceApproval] = {}
    for key, value in raw_cases.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        variant = _string(value.get("variant"))
        case_id = _string(value.get("case_id"))
        if not variant or not case_id:
            continue
        batch_number = _int_value(value.get("batch_number"), fallback=0)
        approvals[key] = EditSourceApproval(
            batch_number=batch_number,
            case_id=case_id,
            variant=variant,
            render_path=Path(_string(value.get("render_path"))),
            score_metric=_string(value.get("score_metric")) or None,
            score=_float_or_none(value.get("score")),
            selected_at=_string(value.get("selected_at")),
        )
    return approvals


def write_edit_source_approvals(
    path: Path,
    approvals: Mapping[str, EditSourceApproval],
) -> None:
    payload: dict[str, JSONValue] = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "cases": {
            key: approval.to_json() for key, approval in sorted(approvals.items())
        },
    }
    write_json_file(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest_paths = _resolve_manifest_paths(args.manifest_glob)
    if not manifest_paths:
        raise SystemExit(f"No manifests matched: {args.manifest_glob}")
    cases = load_selection_cases(
        manifest_paths=manifest_paths,
        create_runs_root=_resolve_path(args.create_runs_root),
        variant_ids=_variant_sequence(args.variants),
    )
    app = EditSourceSelectorApp(
        cases=cases,
        approval_path=_resolve_path(args.approval_path),
    )
    server = _make_server(host=args.host, port=args.port, app=app)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    logger.info("Manual edit source selector: %s", url)
    logger.info("Approval file: %s", app.approval_path)
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


def _make_server(
    *,
    host: str,
    port: int,
    app: EditSourceSelectorApp,
) -> ThreadingHTTPServer:
    class Handler(EditSourceSelectorRequestHandler):
        selector_app = app

    return ThreadingHTTPServer((host, port), Handler)


class EditSourceSelectorRequestHandler(BaseHTTPRequestHandler):
    selector_app: EditSourceSelectorApp

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(
                _HTML.encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return
        if parsed.path == "/api/state":
            self._send_json(self.selector_app.state_json())
            return
        if parsed.path == "/api/image":
            self._handle_image_request(parsed.query)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/approval":
                self._handle_save_approval()
                return
        except (OSError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def log_message(self, format: str, *args: object) -> None:
        logger.debug(format, *args)

    def _handle_save_approval(self) -> None:
        payload = self._read_json_body()
        key = _string(payload.get("key"))
        if not key:
            raise ValueError("key is required.")
        variant = _string(payload.get("variant")) or None
        self.selector_app.save_approval(key=key, variant=variant)
        self._send_json({"ok": True, "state": self.selector_app.state_json()})

    def _handle_image_request(self, query: str) -> None:
        params = parse_qs(query)
        key = params.get("key", [""])[0]
        kind = params.get("kind", ["source"])[0]
        variant = params.get("variant", [""])[0]
        case = self.selector_app.cases_by_key.get(key)
        if case is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown case key.")
            return
        path: Path | None
        if kind == "source":
            path = case.source_image or case.snapshot_image
        elif kind == "render":
            candidate = next(
                (item for item in case.candidates if item.variant == variant),
                None,
            )
            path = candidate.render_path if candidate is not None else None
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
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--manifest-glob", default=_DEFAULT_MANIFEST_GLOB)
    parser.add_argument("--create-runs-root", default=str(_DEFAULT_CREATE_RUNS_ROOT))
    parser.add_argument("--approval-path", default=str(_DEFAULT_APPROVAL_PATH))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(_DEFAULT_SOURCE_VARIANTS),
        help=(
            "Create variants to show as edit-source candidates. Defaults to the "
            "current six core variants in first_time."
        ),
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


def _source_image_for_case(
    *,
    snapshot_image: Path | None,
    candidates: Sequence[EditSourceCandidate],
) -> Path | None:
    for candidate in candidates:
        if candidate.source_image_path is not None:
            return candidate.source_image_path
    return snapshot_image


def _auto_variant(
    *,
    create_batch_root: Path,
    case_id: str,
    candidates: Sequence[EditSourceCandidate],
) -> str | None:
    candidate_variants = {candidate.variant for candidate in candidates}
    configured_path = (
        create_batch_root / case_id / _DEFAULT_CONFIGURED_VARIANT / "render.png"
    )
    best_path = _best_approved_render_path(
        output_root=create_batch_root,
        case_id=case_id,
        configured_path=configured_path,
    )
    if best_path is not None and best_path.parent.name in candidate_variants:
        return best_path.parent.name
    if not candidates:
        return None
    return candidates[0].variant


def _variant_sequence(values: Sequence[str]) -> tuple[str, ...]:
    variants: list[str] = []
    for value in values:
        variants.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(dict.fromkeys(variants))


def _read_metric_items(path: Path) -> tuple[Mapping[str, object], ...]:
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, list):
        return ()
    return tuple(
        cast(Mapping[str, object], item) for item in payload if isinstance(item, dict)
    )


def _score_from_metrics(
    metric_items: Sequence[Mapping[str, object]],
) -> tuple[str | None, float | None, str | None]:
    for metric_name in _SCORE_METRICS:
        metric = _metric_by_name(metric_items, metric_name)
        if metric is None:
            continue
        value = _float_or_none(metric.get("value"))
        if value is not None:
            return metric_name, value, _string(metric.get("status")) or None
    return None, None, None


def _display_metric_summaries(
    metric_items: Sequence[Mapping[str, object]],
) -> tuple[MetricSummary, ...]:
    summaries: list[MetricSummary] = []
    for metric_name in _DISPLAY_METRICS:
        metric = _metric_by_name(metric_items, metric_name)
        if metric is None:
            continue
        summaries.append(
            MetricSummary(
                name=metric_name,
                status=_string(metric.get("status")) or "unknown",
                value=_float_or_none(metric.get("value")),
                threshold=_float_or_none(metric.get("threshold")),
            )
        )
    return tuple(summaries)


def _hard_check_failures(
    metric_items: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    metric = _metric_by_name(metric_items, "layout_fidelity")
    if metric is None:
        return ()
    details = metric.get("details")
    if not isinstance(details, dict):
        return ()
    failures: list[str] = []
    raw_failures = details.get("hard_check_failures")
    if isinstance(raw_failures, list):
        failures.extend(str(item) for item in raw_failures if item)
    raw_checks = details.get("hard_checks")
    if isinstance(raw_checks, dict):
        for name, raw_check in sorted(raw_checks.items()):
            if not isinstance(raw_check, dict) or raw_check.get("status") != "failed":
                continue
            reason = _string(raw_check.get("reason"))
            failures.append(f"{name}: {reason}" if reason else str(name))
    return tuple(dict.fromkeys(failures))


def _layout_reason(metric_items: Sequence[Mapping[str, object]]) -> str | None:
    metric = _metric_by_name(metric_items, "layout_fidelity")
    if metric is None:
        return None
    details = metric.get("details")
    if not isinstance(details, dict):
        return None
    return _string(details.get("reason")) or None


def _metric_by_name(
    metric_items: Sequence[Mapping[str, object]],
    metric_name: str,
) -> Mapping[str, object] | None:
    for item in metric_items:
        if item.get("name") == metric_name:
            return item
    return None


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _existing_path(path: Path) -> Path | None:
    return path if path.is_file() else None


def _is_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_value(value: object, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return fallback
    return fallback


_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TKNT Manual Edit Source Selector</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --ink: #111827;
      --muted: #667085;
      --line: #e5e7eb;
      --soft: #f8fafc;
      --accent: #f97316;
      --green: #16a34a;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f1f5f9; color: var(--ink); }
    header {
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
      padding: 18px 22px; background: white; border-bottom: 1px solid var(--line);
      position: sticky; top: 0; z-index: 5;
    }
    h1 { margin: 0; font-size: 23px; line-height: 1.1; }
    .muted { color: var(--muted); font-size: 13px; }
    .layout { display: grid; grid-template-columns: 360px 1fr; gap: 16px; padding: 16px; }
    .panel { background: white; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
    .case-list { height: calc(100vh - 112px); overflow: auto; padding: 10px; }
    .case-item {
      width: 100%; text-align: left; display: grid; gap: 5px; border: 1px solid var(--line);
      background: var(--soft); color: var(--ink); border-radius: 14px; padding: 12px;
      margin-bottom: 10px; cursor: pointer; font-weight: 800;
    }
    .case-item.active { outline: 3px solid var(--accent); background: #fff7ed; }
    .case-item.manual { border-color: #86efac; background: #f0fdf4; }
    .case-item.missing { opacity: 0.55; }
    .case-meta { color: var(--muted); font-size: 12px; font-weight: 700; }
    .workspace { display: grid; gap: 16px; min-width: 0; }
    .compare { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 16px; }
    .image-card { border: 1px solid var(--line); border-radius: 16px; background: white; overflow: hidden; }
    .image-card h2 { margin: 0; padding: 12px 14px; font-size: 15px; border-bottom: 1px solid var(--line); }
    .image-card img { width: 100%; display: block; background: #e5e7eb; object-fit: contain; max-height: 52vh; }
    .selected-note { padding: 10px 14px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); }
    .candidates { padding: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
    .candidate { border: 1px solid var(--line); border-radius: 16px; background: white; overflow: hidden; display: grid; }
    .candidate.selected { outline: 3px solid var(--green); }
    .candidate.auto:not(.selected) { outline: 2px dashed #f59e0b; }
    .candidate img { width: 100%; aspect-ratio: 3 / 2; object-fit: cover; background: #e5e7eb; }
    .candidate-body { padding: 12px; display: grid; gap: 8px; }
    .candidate-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-weight: 900; }
    .badge { border-radius: 999px; padding: 2px 8px; font-size: 11px; font-weight: 900; color: white; background: var(--muted); }
    .badge.pass { background: var(--green); }
    .badge.fail { background: var(--red); }
    .score { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
    .metrics { display: flex; flex-wrap: wrap; gap: 6px; }
    .metric { border: 1px solid var(--line); border-radius: 999px; padding: 3px 7px; font-size: 11px; color: var(--muted); background: var(--soft); }
    .metric.passed { color: #166534; background: #f0fdf4; border-color: #bbf7d0; }
    .metric.failed { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
    .failures { color: #991b1b; font-size: 12px; display: grid; gap: 3px; }
    button.use {
      border: 0; border-radius: 12px; padding: 10px 12px; font-size: 14px; font-weight: 900;
      background: #111827; color: white; cursor: pointer;
    }
    button.use.selected { background: var(--green); }
    button.clear {
      border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; font-weight: 900;
      background: white; cursor: pointer;
    }
    .status { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TKNT Manual Edit Source Selector</h1>
      <div class="muted">Pick the clean create render that should become the edit source. Scores are shown, but your manual choice wins.</div>
    </div>
    <button class="clear" id="clearButton">Clear Current To Auto</button>
  </header>
  <main class="layout">
    <aside class="panel case-list" id="caseList"></aside>
    <section class="workspace">
      <section class="panel compare">
        <div class="image-card">
          <h2>Source 3D</h2>
          <img id="sourceImage" alt="source 3d" />
        </div>
        <div class="image-card">
          <h2 id="selectedTitle">Selected Render</h2>
          <img id="selectedImage" alt="selected render" />
          <div class="selected-note" id="selectedNote"></div>
        </div>
      </section>
      <section class="panel">
        <div class="candidates" id="candidateGrid"></div>
      </section>
      <section class="panel" style="padding: 14px;">
        <div class="muted">Approval file</div>
        <div class="status" id="statusBox"></div>
      </section>
    </section>
  </main>
  <script>
    const state = { data: null, cases: [], activeKey: null };
    const caseList = document.getElementById('caseList');
    const candidateGrid = document.getElementById('candidateGrid');
    const sourceImage = document.getElementById('sourceImage');
    const selectedImage = document.getElementById('selectedImage');
    const selectedTitle = document.getElementById('selectedTitle');
    const selectedNote = document.getElementById('selectedNote');
    const statusBox = document.getElementById('statusBox');
    const clearButton = document.getElementById('clearButton');

    function currentCase() {
      return state.cases.find((item) => item.key === state.activeKey) ?? null;
    }
    async function loadState() {
      const response = await fetch('/api/state');
      state.data = await response.json();
      state.cases = state.data.cases ?? [];
      if (!state.activeKey && state.cases.length > 0) state.activeKey = state.cases[0].key;
      renderCaseList();
      renderActiveCase();
      statusBox.textContent = state.data.approval_path ?? '';
    }
    function renderCaseList() {
      caseList.innerHTML = '';
      for (const item of state.cases) {
        const selected = item.selected_variant ?? 'none';
        const button = document.createElement('button');
        button.className = [
          'case-item',
          item.key === state.activeKey ? 'active' : '',
          item.selection_manual ? 'manual' : '',
          item.candidates.length ? '' : 'missing',
        ].join(' ');
        button.innerHTML = '';
        const title = document.createElement('span');
        title.textContent = `B${item.batch_number} · ${item.case_id}`;
        const meta = document.createElement('span');
        meta.className = 'case-meta';
        meta.textContent = `${item.selection_manual ? 'manual' : 'auto'}: ${selected}`;
        button.appendChild(title);
        button.appendChild(meta);
        button.onclick = () => {
          state.activeKey = item.key;
          renderCaseList();
          renderActiveCase();
        };
        caseList.appendChild(button);
      }
    }
    function renderActiveCase() {
      const item = currentCase();
      if (!item) return;
      sourceImage.src = item.source_url + '&t=' + Date.now();
      const selected = candidateByVariant(item, item.selected_variant) ?? item.candidates[0] ?? null;
      if (selected) {
        selectedImage.src = renderUrl(item, selected.variant);
        selectedTitle.textContent = `Selected Render · ${selected.variant}`;
        selectedNote.textContent = `${item.selection_manual ? 'Manual exact' : 'Auto'} · ${scoreLabel(selected)} · ${selected.render_path}`;
      } else {
        selectedImage.removeAttribute('src');
        selectedTitle.textContent = 'Selected Render';
        selectedNote.textContent = 'No render candidates found.';
      }
      renderCandidates(item);
    }
    function renderCandidates(item) {
      candidateGrid.innerHTML = '';
      for (const candidate of item.candidates) {
        const selected = candidate.variant === item.selected_variant;
        const auto = candidate.variant === item.auto_variant;
        const card = document.createElement('article');
        card.className = ['candidate', selected ? 'selected' : '', auto ? 'auto' : ''].join(' ');
        const image = document.createElement('img');
        image.src = renderUrl(item, candidate.variant);
        image.alt = candidate.variant;
        image.onclick = () => saveApproval(item.key, candidate.variant);
        const body = document.createElement('div');
        body.className = 'candidate-body';
        const title = document.createElement('div');
        title.className = 'candidate-title';
        const name = document.createElement('span');
        name.textContent = candidate.variant;
        const badge = document.createElement('span');
        badge.className = `badge ${candidate.passed ? 'pass' : 'fail'}`;
        badge.textContent = candidate.passed ? 'passed' : (candidate.score_status ?? 'no score');
        title.appendChild(name);
        title.appendChild(badge);
        const score = document.createElement('div');
        score.className = 'score';
        score.textContent = scoreLabel(candidate);
        const metrics = document.createElement('div');
        metrics.className = 'metrics';
        for (const metric of candidate.metrics) {
          const pill = document.createElement('span');
          pill.className = `metric ${metric.status}`;
          pill.textContent = `${shortMetric(metric.name)}=${formatValue(metric.value)} ${metric.status}`;
          metrics.appendChild(pill);
        }
        const failures = document.createElement('div');
        failures.className = 'failures';
        for (const failure of candidate.hard_check_failures.slice(0, 4)) {
          const row = document.createElement('span');
          row.textContent = `hard fail: ${failure}`;
          failures.appendChild(row);
        }
        if (candidate.reason) {
          const reason = document.createElement('div');
          reason.className = 'muted';
          reason.textContent = candidate.reason;
          failures.appendChild(reason);
        }
        const button = document.createElement('button');
        button.className = `use ${selected ? 'selected' : ''}`;
        button.textContent = selected ? 'Using for edit' : 'Use for edit';
        button.onclick = () => saveApproval(item.key, candidate.variant);
        body.appendChild(title);
        body.appendChild(score);
        body.appendChild(metrics);
        body.appendChild(failures);
        body.appendChild(button);
        card.appendChild(image);
        card.appendChild(body);
        candidateGrid.appendChild(card);
      }
    }
    function candidateByVariant(item, variant) {
      return item.candidates.find((candidate) => candidate.variant === variant) ?? null;
    }
    function renderUrl(item, variant) {
      return `/api/image?key=${encodeURIComponent(item.key)}&kind=render&variant=${encodeURIComponent(variant)}&t=${Date.now()}`;
    }
    function scoreLabel(candidate) {
      if (candidate.score == null) return 'score: missing';
      return `${candidate.score_metric ?? 'score'}=${Number(candidate.score).toFixed(3)} (${candidate.score_status ?? 'unknown'})`;
    }
    function formatValue(value) {
      return value == null ? 'na' : Number(value).toFixed(3);
    }
    function shortMetric(name) {
      return name
        .replace('layout_', 'l_')
        .replace('detector_', 'det_')
        .replace('_similarity', '_sim')
        .replace('_structure', '_struct')
        .replace('_score', '');
    }
    async function saveApproval(key, variant) {
      const response = await fetch('/api/approval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, variant }),
      });
      const payload = await response.json();
      if (!payload.ok) {
        statusBox.textContent = payload.error ?? 'save failed';
        return;
      }
      state.data = payload.state;
      state.cases = state.data.cases ?? [];
      renderCaseList();
      renderActiveCase();
      statusBox.textContent = `${state.data.approval_path}\nSaved ${variant || 'auto'} for ${key}`;
    }
    clearButton.onclick = () => {
      const item = currentCase();
      if (item) saveApproval(item.key, null);
    };
    window.addEventListener('load', loadState);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
