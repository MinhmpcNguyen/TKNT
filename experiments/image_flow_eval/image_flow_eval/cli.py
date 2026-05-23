from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from .io import load_manifest, metric_results_to_json, write_json_file, write_metric_csv
from .metrics import compute_fid_score, evaluate_variant_images, summarize_human_ratings
from .providers import resolve_provider, run_case_variant
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
    "no_layout_reference",
    "no_edit_mask",
    "no_preservation_prompt",
    "no_reference_images",
)


def main() -> int:
    args = _parse_args()
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "human-summary":
        return _human_summary_command(args)
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
        choices=("tknt_snapshot", "command", "evaluate_only"),
        default="tknt_snapshot",
    )
    run.add_argument(
        "--variants",
        nargs="*",
        choices=_ALL_ABLATION_MODES,
        help="Ablation modes to run. Defaults to manifest variants.",
    )

    human = subparsers.add_parser(
        "human-summary",
        help="Aggregate human rating CSV into a JSON summary.",
    )
    human.add_argument("--ratings", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)
    human.add_argument("--artifact-fail-threshold", type=float, default=3.0)
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
        generated_images=[
            Path(str(row["output_image"]))
            for row in rows
            if isinstance(row.get("output_image"), str)
        ],
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
    rows: list[dict[str, JSONValue]] = []
    failed = False
    generated_images: list[Path] = []

    for case in manifest.cases:
        case_variants = variants or case.variants
        for variant in case_variants:
            provider = resolve_provider(default_provider, variant)
            provider_result, provider_error = _safe_run_provider(
                provider=provider,
                case=case,
                variant=variant,
                output_dir=output_dir,
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
                _write_variant_metrics(output_dir, case.id, variant.id, [provider_error])
                continue

            assert provider_result is not None
            generated_images.append(provider_result.output_image)
            metrics = evaluate_variant_images(
                manifest=manifest,
                case=case,
                variant=variant,
                output_image=provider_result.output_image,
                output_dir=output_dir,
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
        generated_images=generated_images,
    )
    return 1 if failed else 0


def _human_summary_command(args: argparse.Namespace) -> int:
    summary = summarize_human_ratings(
        ratings_csv=args.ratings,
        artifact_fail_threshold=args.artifact_fail_threshold,
    )
    write_json_file(args.output, summary)
    return 0


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
) -> tuple[ProviderResult | None, MetricResult | None]:
    try:
        return (
            run_case_variant(
                provider=provider,
                case=case,
                variant=variant,
                output_root=output_dir,
            ),
            None,
        )
    except Exception as exc:
        return (
            None,
            MetricResult(
                name="provider_run",
                status="failed",
                details={
                    "provider": provider,
                    "reason": str(exc),
                },
            ),
        )


def _write_reports(
    *,
    manifest: ManifestConfig,
    output_dir: Path,
    result_rows: list[dict[str, JSONValue]],
    generated_images: list[Path],
) -> None:
    batch_metrics: list[MetricResult] = []
    if manifest.fid_reference_images:
        batch_metrics.append(
            compute_fid_score(
                reference_images=manifest.fid_reference_images,
                generated_images=generated_images,
            )
        )
    write_json_file(
        output_dir / "summary.json",
        {
            "manifest": str(manifest.path),
            "output_dir": str(output_dir),
            "results": result_rows,
            "batch_metrics": metric_results_to_json(batch_metrics),
        },
    )
    csv_rows: list[dict[str, JSONValue]] = []
    for row in result_rows:
        case_id = str(row["case_id"])
        variant_id = str(row["variant_id"])
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


if __name__ == "__main__":
    raise SystemExit(main())
