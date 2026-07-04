from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias, cast

from experiments.image_flow_eval.image_flow_eval.io import (
    load_manifest,
    metric_results_to_json,
    write_json_file,
    write_metric_csv,
)
from experiments.image_flow_eval.image_flow_eval.metrics import (
    compute_layout_gate_score,
    evaluate_variant_images,
)
from experiments.image_flow_eval.image_flow_eval.providers import run_case_variant
from experiments.image_flow_eval.image_flow_eval.types import (
    CaseConfig,
    JSONValue,
    ManifestConfig,
    MetricDirection,
    MetricResult,
    MetricStatus,
    VariantConfig,
)

logger = logging.getLogger(__name__)

MetricPayload: TypeAlias = dict[str, JSONValue]
_RATE_LIMIT_PATTERN = re.compile(
    r"(?i)(429\s+too\s+many\s+requests?|rate.?limit|too\s+many\s+requests)"
)
_SETUP_FAILURE_PATTERN = re.compile(
    r"(?i)(api_key_missing|No module named ['\"]config['\"]|config_error)"
)
_BATCH_PATTERN = re.compile(r"batch(?P<number>\d+)$")
_DEPENDENT_METRICS = {"layout_gate_score"}
_REPORT_FILENAMES = (
    "layout_fidelity_report.json",
    "preset_fidelity_report.json",
)


@dataclass(frozen=True)
class RetryTarget:
    run_root: Path
    batch_dir: Path
    case_id: str
    variant_id: str
    failed_metric_names: frozenset[str]

    @property
    def metrics_path(self) -> Path:
        return self.batch_dir / self.case_id / self.variant_id / "metrics.json"

    @property
    def variant_dir(self) -> Path:
        return self.metrics_path.parent


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _configure_metric_environment(Path(__file__).resolve().parents[2])
    run_roots = tuple(path.expanduser().resolve() for path in args.run_root)
    manifest_dir = args.manifest_dir.expanduser().resolve()
    targets = _discover_targets(run_roots)
    _log_targets(targets)
    if args.dry_run or not targets:
        return 0
    _require_evaluator_api_key()

    manifests: dict[Path, ManifestConfig] = {}
    updated_batch_dirs: set[Path] = set()
    success_count = 0
    failure_count = 0
    for index, target in enumerate(targets, start=1):
        manifest_path = _manifest_path_for_batch(
            target.batch_dir,
            manifest_dir=manifest_dir,
        )
        manifest = manifests.setdefault(manifest_path, load_manifest(manifest_path))
        case, variant = _resolve_case_variant(
            manifest,
            case_id=target.case_id,
            variant_id=target.variant_id,
        )
        logger.info(
            "[%d/%d] Re-evaluating %s / %s / %s",
            index,
            len(targets),
            target.batch_dir.name,
            target.case_id,
            target.variant_id,
        )
        updated = _retry_target(
            target=target,
            manifest=manifest,
            case=case,
            variant=variant,
            max_attempts=args.max_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
        )
        if updated:
            success_count += 1
            updated_batch_dirs.add(target.batch_dir)
        else:
            failure_count += 1
        if index < len(targets) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    for batch_dir in sorted(updated_batch_dirs):
        _refresh_summary(batch_dir)

    logger.info(
        "Completed metric-only retry: %d updated, %d unresolved.",
        success_count,
        failure_count,
    )
    return 1 if failure_count else 0


def _configure_metric_environment(repo_root: Path) -> None:
    cache_root = repo_root / ".cache"
    cache_defaults = {
        "TORCH_HOME": cache_root / "torch",
        "HF_HOME": cache_root / "huggingface",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "YOLO_CONFIG_DIR": cache_root / "ultralytics",
    }
    for name, path in cache_defaults.items():
        os.environ.setdefault(name, str(path))
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "TKNT_IMAGE_FLOW_YOLO_MODEL",
        str(repo_root / "models/metrics/yolo11n.pt"),
    )
    os.environ.setdefault("TKNT_IMAGE_FLOW_METRIC_DEVICE", "cpu")
    os.environ.setdefault("TKNT_IMAGE_FLOW_DINO_MODEL", "facebook/dinov2-small")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run only existing image-flow metric rows whose metrics.json "
            "contains an HTTP 429 evaluator failure."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
        help="Run root to scan. Repeat this option for multiple roots.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("experiments/image_flow_eval/snapshot_audit"),
    )
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--initial-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _discover_targets(run_roots: Sequence[Path]) -> list[RetryTarget]:
    targets: list[RetryTarget] = []
    for run_root in run_roots:
        for metrics_path in sorted(run_root.glob("*/*/*/metrics.json")):
            metrics = _read_metric_payloads(metrics_path)
            failed_names = frozenset(
                name
                for metric in metrics
                if _contains_retryable_evaluator_failure(metric)
                if (name := _metric_name(metric))
            )
            if not failed_names:
                continue
            relative = metrics_path.relative_to(run_root)
            targets.append(
                RetryTarget(
                    run_root=run_root,
                    batch_dir=run_root / relative.parts[0],
                    case_id=relative.parts[1],
                    variant_id=relative.parts[2],
                    failed_metric_names=failed_names,
                )
            )
    return targets


def _retry_target(
    *,
    target: RetryTarget,
    manifest: ManifestConfig,
    case: CaseConfig,
    variant: VariantConfig,
    max_attempts: int,
    initial_backoff_seconds: float,
) -> bool:
    original_metrics = _read_metric_payloads(target.metrics_path)
    report_backups = _backup_reports(target.variant_dir)
    backoff_seconds = max(0.0, initial_backoff_seconds)
    for attempt in range(1, max(1, max_attempts) + 1):
        provider_result = run_case_variant(
            provider="evaluate_only",
            case=case,
            variant=variant,
            output_root=target.batch_dir,
        )
        judge_manifest = replace(
            manifest,
            metrics=tuple(sorted(target.failed_metric_names)),
        )
        results = evaluate_variant_images(
            manifest=judge_manifest,
            case=case,
            variant=variant,
            output_image=provider_result.output_image,
            output_dir=target.batch_dir,
            provider_result=provider_result,
        )
        new_metrics = cast(
            list[MetricPayload],
            metric_results_to_json(results),
        )
        if any(_contains_rate_limit(metric) for metric in new_metrics):
            logger.warning(
                "429 persisted on attempt %d/%d; waiting %.1fs.",
                attempt,
                max_attempts,
                backoff_seconds,
            )
            if attempt < max_attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds)
                backoff_seconds = max(backoff_seconds * 2, backoff_seconds + 1)
            continue
        if any(_contains_transport_failure(metric) for metric in new_metrics):
            logger.error(
                "A non-429 transport failure occurred; preserving original metrics."
            )
            _restore_reports(target.variant_dir, report_backups)
            return False

        judge_metric_names = set(target.failed_metric_names)
        metrics_with_new_judges = _merge_metrics(
            original_metrics,
            new_metrics,
            replacement_names=judge_metric_names,
        )
        gate_result = compute_layout_gate_score(
            metrics=[
                _metric_result_from_payload(metric)
                for metric in metrics_with_new_judges
            ],
            thresholds=manifest.thresholds,
        )
        gate_payload = cast(MetricPayload, gate_result.to_json())
        replacement_names = judge_metric_names | _DEPENDENT_METRICS
        merged_metrics = _merge_metrics(
            metrics_with_new_judges,
            [gate_payload],
            replacement_names=_DEPENDENT_METRICS,
        )
        write_json_file(target.metrics_path, merged_metrics)
        logger.info(
            "Updated metrics: %s",
            ", ".join(sorted(replacement_names)),
        )
        return True

    _restore_reports(target.variant_dir, report_backups)
    logger.error("429 remained after %d attempt(s); original metrics preserved.", max_attempts)
    return False


def _merge_metrics(
    original: Sequence[MetricPayload],
    replacement: Sequence[MetricPayload],
    *,
    replacement_names: set[str],
) -> list[MetricPayload]:
    replacement_by_name = {
        name: metric
        for metric in replacement
        if (name := _metric_name(metric)) in replacement_names
    }
    merged: list[MetricPayload] = []
    seen: set[str] = set()
    for metric in original:
        name = _metric_name(metric)
        if name in replacement_by_name:
            merged.append(replacement_by_name[name])
            seen.add(name)
        else:
            merged.append(metric)
    for name in sorted(replacement_names - seen):
        metric = replacement_by_name.get(name)
        if metric is not None:
            merged.append(metric)
    return merged


def _metric_result_from_payload(metric: Mapping[str, object]) -> MetricResult:
    status_value = str(metric.get("status") or "failed")
    status = cast(
        MetricStatus,
        status_value
        if status_value in {"passed", "failed", "warning", "skipped"}
        else "failed",
    )
    direction_value = str(metric.get("direction") or "none")
    direction = cast(
        MetricDirection,
        direction_value
        if direction_value in {"min", "max", "none"}
        else "none",
    )
    details = metric.get("details")
    return MetricResult(
        name=_metric_name(metric),
        status=status,
        value=_optional_float(metric.get("value")),
        threshold=_optional_float(metric.get("threshold")),
        direction=direction,
        details=cast(Mapping[str, JSONValue], details)
        if isinstance(details, dict)
        else {},
    )


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _manifest_path_for_batch(batch_dir: Path, *, manifest_dir: Path) -> Path:
    match = _BATCH_PATTERN.search(batch_dir.name)
    if match is None:
        raise ValueError(f"Unable to determine batch number from: {batch_dir}")
    path = manifest_dir / (
        f"manifest.curated_core14_create.batch{match.group('number')}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path


def _resolve_case_variant(
    manifest: ManifestConfig,
    *,
    case_id: str,
    variant_id: str,
) -> tuple[CaseConfig, VariantConfig]:
    case = next((item for item in manifest.cases if item.id == case_id), None)
    if case is None:
        raise ValueError(f"Case {case_id} not found in {manifest.path}")
    variant = next((item for item in case.variants if item.id == variant_id), None)
    if variant is None:
        raise ValueError(
            f"Variant {variant_id} not found for case {case_id} in {manifest.path}"
        )
    return case, variant


def _backup_reports(variant_dir: Path) -> dict[str, bytes | None]:
    return {
        filename: (path.read_bytes() if path.is_file() else None)
        for filename in _REPORT_FILENAMES
        if (path := variant_dir / filename)
    }


def _restore_reports(
    variant_dir: Path,
    backups: Mapping[str, bytes | None],
) -> None:
    for filename, content in backups.items():
        path = variant_dir / filename
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def _refresh_summary(batch_dir: Path) -> None:
    summary_path = batch_dir / "summary.json"
    if not summary_path.is_file():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        logger.warning("Skipping malformed summary: %s", summary_path)
        return
    results = cast(list[object], payload["results"])
    csv_rows: list[dict[str, JSONValue]] = []
    for raw_row in results:
        if not isinstance(raw_row, dict):
            continue
        row = cast(dict[str, object], raw_row)
        case_id = str(row.get("case_id") or "")
        variant_id = str(row.get("variant_id") or "")
        metrics_path = batch_dir / case_id / variant_id / "metrics.json"
        if metrics_path.is_file():
            metrics = _read_metric_payloads(metrics_path)
            row["metrics"] = metrics
            row["aggregate_status"] = _aggregate_status(metrics)
        metrics = row.get("metrics")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            csv_rows.append(
                {
                    "case_id": case_id,
                    "variant_id": variant_id,
                    "provider_status": str(row.get("provider_status") or ""),
                    "metric": str(metric.get("name") or ""),
                    "status": str(metric.get("status") or ""),
                    "value": cast(JSONValue, metric.get("value")),
                    "threshold": cast(JSONValue, metric.get("threshold")),
                    "direction": str(metric.get("direction") or ""),
                    "details": cast(JSONValue, metric.get("details") or {}),
                }
            )
    write_json_file(summary_path, cast(dict[str, JSONValue], payload))
    write_metric_csv(batch_dir / "summary.csv", csv_rows)
    logger.info("Refreshed summaries: %s", batch_dir)


def _aggregate_status(metrics: Sequence[MetricPayload]) -> str:
    statuses = {str(metric.get("status") or "") for metric in metrics}
    if "failed" in statuses:
        return "failed"
    if statuses.intersection({"warning", "skipped"}):
        return "warning"
    return "passed"


def _read_metric_payloads(path: Path) -> list[MetricPayload]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else []
    return [cast(MetricPayload, item) for item in items if isinstance(item, dict)]


def _metric_name(metric: Mapping[str, object]) -> str:
    value = metric.get("name") or metric.get("metric")
    return value if isinstance(value, str) else ""


def _contains_rate_limit(value: object) -> bool:
    return _RATE_LIMIT_PATTERN.search(json.dumps(value, ensure_ascii=False)) is not None


def _contains_retryable_evaluator_failure(metric: Mapping[str, object]) -> bool:
    if _metric_name(metric) not in {"layout_fidelity", "preset_fidelity"}:
        return False
    serialized = json.dumps(metric, ensure_ascii=False)
    return bool(
        _RATE_LIMIT_PATTERN.search(serialized)
        or _SETUP_FAILURE_PATTERN.search(serialized)
    )


def _contains_transport_failure(metric: Mapping[str, object]) -> bool:
    details = metric.get("details")
    if not isinstance(details, dict):
        return False
    reason = str(details.get("reason") or "")
    error_type = str(details.get("error_type") or "")
    if reason in {
        "layout_fidelity_vision_api_key_missing",
        "preset_fidelity_vision_api_key_missing",
        "vision_api_key_missing",
    }:
        return True
    return reason in {
        "layout_fidelity_vision_eval_failed",
        "preset_fidelity_vision_eval_failed",
    } and bool(error_type)


def _require_evaluator_api_key() -> None:
    if any(
        os.getenv(name, "").strip()
        for name in ("OPENAI_API_KEY", "OPENAI_IMAGE_API_KEY")
    ):
        return
    try:
        from config.openai_config import OpenAIConfig
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import config.openai_config from the repository root. "
            "Run this utility with `python -m experiments.image_flow_eval."
            "rerun_429_metrics`."
        ) from exc
    if not OpenAIConfig.OPENAI_API_KEY:
        raise RuntimeError(
            "Evaluator API key is unavailable. Configure OPENAI_API_KEY or "
            "OPENAI_IMAGE_API_KEY before retrying metrics."
        )


def _log_targets(targets: Sequence[RetryTarget]) -> None:
    logger.info(
        "Found %d variant row(s) with retryable evaluator failures.",
        len(targets),
    )
    grouped: dict[tuple[str, str], int] = {}
    for target in targets:
        key = (target.run_root.name, target.batch_dir.name)
        grouped[key] = grouped.get(key, 0) + 1
    for (root_name, batch_name), count in sorted(grouped.items()):
        logger.info("  %s/%s: %d row(s)", root_name, batch_name, count)


if __name__ == "__main__":
    raise SystemExit(main())
