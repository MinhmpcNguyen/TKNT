from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import cast

from .io import load_manifest, metric_results_to_json, write_json_file, write_metric_csv
from .metrics import evaluate_variant_images, summarize_human_ratings
from .providers import resolve_provider, run_case_variant
from .semantic_repair import run_semantic_repair
from .types import (
    CaseConfig,
    JSONValue,
    ManifestConfig,
    MetricResult,
    ProviderName,
    ProviderResult,
    VariantConfig,
    VariantMode,
)

_ALL_ABLATION_MODES: tuple[VariantMode, ...] = (
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
)


def main() -> int:
    args = _parse_args()
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "human-summary":
        return _human_summary_command(args)
    if args.command == "repair-semantic":
        return _repair_semantic_command(args)
    raise ValueError(f"Unknown command: {args.command}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TKNT image render/edit flow outputs and ablations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate existing variant output images from a manifest.",
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)

    run = subparsers.add_parser(
        "run",
        help="Run image variants through a provider, then evaluate outputs.",
    )
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--provider",
        choices=("tknt_snapshot", "evaluate_only"),
        default="tknt_snapshot",
    )
    run.add_argument(
        "--variants",
        nargs="*",
        choices=_ALL_ABLATION_MODES,
        help="Ablation modes to run. Defaults to manifest variants.",
    )
    run.add_argument(
        "--resume",
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Reuse completed variant images and resume partial edit variants.",
    )

    human = subparsers.add_parser(
        "human-summary",
        help="Aggregate human rating CSV into a JSON summary.",
    )
    human.add_argument("--ratings", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)
    human.add_argument("--artifact-fail-threshold", type=float, default=3.0)

    repair = subparsers.add_parser(
        "repair-semantic",
        help="Repair existing single_edit_edge_silhouette_guided outputs with an LLM-planned edit pass.",
    )
    repair.add_argument("--run-dir", type=Path, required=True)
    repair.add_argument("--output-dir", type=Path, required=True)
    repair.add_argument(
        "--source-variant",
        default="single_edit_edge_silhouette_guided",
        help="Existing source variant folder to repair.",
    )
    repair.add_argument(
        "--output-variant",
        default="single_edit_edge_silhouette_guided_llm_repair",
        help="Variant folder name to write under each case.",
    )
    repair.add_argument(
        "--cases",
        nargs="*",
        default=(),
        help="Optional case ids to repair. Defaults to all reusable cases in --run-dir.",
    )
    repair.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing repair plans and completed repair renders.",
    )
    repair.add_argument(
        "--plan-only",
        action="store_true",
        help="Only ask the LLM for repair plans; do not call the image edit model.",
    )
    repair.add_argument(
        "--force-repair",
        action="store_true",
        help="Run the image repair pass even when the LLM planner says no repair is needed.",
    )
    repair.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate repair outputs even if --resume would otherwise reuse them.",
    )
    repair.add_argument(
        "--refresh-plan",
        action="store_true",
        help="Ask the LLM planner again instead of reusing an existing repair plan.",
    )
    repair.add_argument(
        "--enhanced-geometry-guide",
        action="store_true",
        help="Build and pass a deterministic sharpened edge/object/crop guide as the structural reference.",
    )
    repair.add_argument(
        "--guide-only",
        action="store_true",
        help="Only build/copy repair inputs and the deterministic guide; do not call LLM or image edit.",
    )
    return parser.parse_args()


def _evaluate_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    output_dir = _resolve_output_dir(
        manifest=manifest,
        requested=args.output_dir,
        command_name="evaluate",
    )
    rows, failed = _evaluate_existing_outputs(manifest=manifest, output_dir=output_dir)
    _write_reports(
        manifest=manifest,
        output_dir=output_dir,
        result_rows=rows,
    )
    return 1 if failed else 0


def _run_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    output_dir = _resolve_output_dir(
        manifest=manifest,
        requested=args.output_dir,
        command_name="run",
    )
    default_provider = cast(ProviderName, args.provider)
    variants = _cli_variants(args.variants)
    skip_existing = bool(args.skip_existing)
    rows: list[dict[str, JSONValue]] = []
    failed = False

    for case in manifest.cases:
        case_variants = variants or case.variants
        for variant in case_variants:
            provider = resolve_provider(default_provider, variant)
            provider_result, provider_error = _safe_run_provider(
                provider=provider,
                case=case,
                variant=variant,
                output_dir=output_dir,
                skip_existing=skip_existing,
            )
            if provider_error is not None:
                failed = True
                row = _result_row(
                    case=case,
                    variant=variant,
                    output_image=None,
                    provider_result=None,
                    metrics=[provider_error],
                )
                rows.append(row)
                _write_variant_metrics(
                    output_dir, case.id, variant.id, [provider_error]
                )
                continue

            assert provider_result is not None
            metrics = evaluate_variant_images(
                manifest=manifest,
                case=case,
                variant=variant,
                output_image=provider_result.output_image,
                output_dir=output_dir,
                provider_result=provider_result,
            )
            if any(metric.status == "failed" for metric in metrics):
                failed = True
            rows.append(
                _result_row(
                    case=case,
                    variant=variant,
                    output_image=provider_result.output_image,
                    provider_result=provider_result,
                    metrics=metrics,
                )
            )
            _write_variant_metrics(output_dir, case.id, variant.id, metrics)

    _write_reports(
        manifest=manifest,
        output_dir=output_dir,
        result_rows=rows,
    )
    return 1 if failed else 0


def _human_summary_command(args: argparse.Namespace) -> int:
    summary = summarize_human_ratings(
        ratings_csv=args.ratings,
        artifact_fail_threshold=args.artifact_fail_threshold,
    )
    write_json_file(args.output, summary)
    return 0


def _repair_semantic_command(args: argparse.Namespace) -> int:
    results = run_semantic_repair(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        source_variant=str(args.source_variant),
        output_variant=str(args.output_variant),
        case_ids=tuple(str(case_id) for case_id in args.cases),
        resume=bool(args.resume),
        plan_only=bool(args.plan_only),
        force_repair=bool(args.force_repair),
        overwrite=bool(args.overwrite),
        refresh_plan=bool(args.refresh_plan),
        enhanced_geometry_guide=bool(args.enhanced_geometry_guide),
        guide_only=bool(args.guide_only),
    )
    return 1 if any(result.status == "failed" for result in results) else 0


def _evaluate_existing_outputs(
    *,
    manifest: ManifestConfig,
    output_dir: Path,
) -> tuple[list[dict[str, JSONValue]], bool]:
    rows: list[dict[str, JSONValue]] = []
    failed = False
    for case in manifest.cases:
        for variant in case.variants:
            provider_result, provider_error = _safe_run_provider(
                provider="evaluate_only",
                case=case,
                variant=variant,
                output_dir=output_dir,
            )
            if provider_error is not None:
                failed = True
                metrics = [provider_error]
                rows.append(
                    _result_row(
                        case=case,
                        variant=variant,
                        output_image=None,
                        provider_result=None,
                        metrics=metrics,
                    )
                )
                _write_variant_metrics(output_dir, case.id, variant.id, metrics)
                continue

            assert provider_result is not None
            metrics = evaluate_variant_images(
                manifest=manifest,
                case=case,
                variant=variant,
                output_image=provider_result.output_image,
                output_dir=output_dir,
                provider_result=provider_result,
            )
            if any(metric.status == "failed" for metric in metrics):
                failed = True
            rows.append(
                _result_row(
                    case=case,
                    variant=variant,
                    output_image=provider_result.output_image,
                    provider_result=provider_result,
                    metrics=metrics,
                )
            )
            _write_variant_metrics(output_dir, case.id, variant.id, metrics)
    return rows, failed


def _safe_run_provider(
    *,
    provider: ProviderName,
    case: CaseConfig,
    variant: VariantConfig,
    output_dir: Path,
    skip_existing: bool = False,
) -> tuple[ProviderResult | None, MetricResult | None]:
    started_at = perf_counter()
    try:
        result = run_case_variant(
            provider=provider,
            case=case,
            variant=variant,
            output_root=output_dir,
            skip_existing=skip_existing,
        )
        runtime_s = perf_counter() - started_at
        _write_provider_runtime_metadata(result.metadata_path, runtime_s=runtime_s)
        return (
            replace(result, runtime_s=runtime_s),
            None,
        )
    except Exception as exc:
        runtime_s = perf_counter() - started_at
        return (
            None,
            MetricResult(
                name="provider_run",
                status="failed",
                details={
                    "provider": provider,
                    "reason": str(exc),
                    "runtime_s": round(runtime_s, 6),
                },
            ),
        )


def _write_reports(
    *,
    manifest: ManifestConfig,
    output_dir: Path,
    result_rows: list[dict[str, JSONValue]],
) -> None:
    write_json_file(
        output_dir / "summary.json",
        {
            "manifest": str(manifest.path),
            "output_dir": str(output_dir),
            "results": result_rows,
        },
    )
    csv_rows: list[dict[str, JSONValue]] = []
    for row in result_rows:
        case_id = str(row["case_id"])
        variant_id = str(row["variant_id"])
        provider_status = str(row.get("provider_status", ""))
        metrics = row.get("metrics")
        if not isinstance(metrics, list):
            continue
        for item in metrics:
            if not isinstance(item, dict):
                continue
            csv_rows.append(
                {
                    "case_id": case_id,
                    "variant_id": variant_id,
                    "provider_status": provider_status,
                    "metric": str(item.get("name", "")),
                    "status": str(item.get("status", "")),
                    "value": item.get("value"),
                    "threshold": item.get("threshold"),
                    "direction": str(item.get("direction", "")),
                    "details": cast(JSONValue, item.get("details", {})),
                }
            )
    write_metric_csv(output_dir / "summary.csv", csv_rows)


def _write_variant_metrics(
    output_dir: Path,
    case_id: str,
    variant_id: str,
    metrics: Sequence[MetricResult],
) -> None:
    write_json_file(
        output_dir / case_id / variant_id / "metrics.json",
        metric_results_to_json(metrics),
    )


def _result_row(
    *,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path | None,
    provider_result: ProviderResult | None,
    metrics: Sequence[MetricResult],
) -> dict[str, JSONValue]:
    return {
        "case_id": case.id,
        "variant_id": variant.id,
        "mode": variant.mode,
        "output_image": str(output_image) if output_image is not None else None,
        "provider_metadata": str(provider_result.metadata_path)
        if provider_result and provider_result.metadata_path
        else None,
        "render_image": str(provider_result.render_image)
        if provider_result and provider_result.render_image
        else None,
        "source_image": str(provider_result.source_image)
        if provider_result and provider_result.source_image
        else None,
        "source_snapshot_payload": str(provider_result.source_snapshot_payload)
        if provider_result and provider_result.source_snapshot_payload
        else None,
        "approved_render_source": str(provider_result.approved_render_source)
        if provider_result and provider_result.approved_render_source
        else None,
        "provider_status": provider_result.provider_status
        if provider_result
        else "generated",
        "provider_runtime_s": round(provider_result.runtime_s, 6)
        if provider_result and provider_result.runtime_s is not None
        else None,
        "aggregate_status": _aggregate_status(metrics),
        "metrics": metric_results_to_json(metrics),
    }


def _aggregate_status(metrics: Sequence[MetricResult]) -> str:
    if any(metric.status == "failed" for metric in metrics):
        return "failed"
    if any(metric.status in {"warning", "skipped"} for metric in metrics):
        return "warning"
    return "passed"


def _resolve_output_dir(
    *,
    manifest: ManifestConfig,
    requested: Path | None,
    command_name: str,
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if manifest.output_dir is not None:
        return manifest.output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (manifest.path.parent / "runs" / f"{command_name}_{stamp}").resolve()


def _cli_variants(values: list[str] | None) -> tuple[VariantConfig, ...] | None:
    if not values:
        return None
    return tuple(
        VariantConfig(
            id=value,
            mode=cast(VariantMode, value),
        )
        for value in values
    )


def _write_provider_runtime_metadata(
    metadata_path: Path | None,
    *,
    runtime_s: float,
) -> None:
    if metadata_path is None or not metadata_path.is_file():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["provider_runtime_s"] = round(runtime_s, 6)
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
