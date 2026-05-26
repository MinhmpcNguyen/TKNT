from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from evaluation.layout_benchmark.external_runners import run_external_case
from evaluation.layout_benchmark.normalize import load_output_payload, normalize_payload
from evaluation.layout_benchmark.schema import normalized_layout_to_json
from evaluation.layout_benchmark.scoring import canonical_object_type, score_layout

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_MODEL = "gpt-5.4-mini"
WRITE_STDOUT_LOG_ENV = "LAYOUT_BENCHMARK_WRITE_STDOUT_LOG"
SYSTEM_ID_ALIASES: Mapping[str, str] = {
    "idesign_gemini": "idesign_openai",
    "flairgpt_gemini": "flairgpt_openai",
}


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    kind: str
    normalizer: str
    ablation: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan, run, and score the TKNT layout benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "run", "score"),
        default="plan",
        help="plan writes the schedule, run executes TKNT systems, score scores existing outputs.",
    )
    parser.add_argument(
        "--cases",
        default=str(BENCHMARK_DIR / "cases.json"),
        help="Benchmark cases JSON.",
    )
    parser.add_argument(
        "--systems",
        default=str(BENCHMARK_DIR / "systems.json"),
        help="Benchmark systems JSON.",
    )
    parser.add_argument(
        "--output-root",
        default=str(BENCHMARK_DIR / "runs" / "latest"),
        help="Folder for plan, outputs, normalized layouts, and scores.",
    )
    parser.add_argument(
        "--external-root",
        default=str(BENCHMARK_DIR / "external_outputs"),
        help="Folder containing IDesign/FlairGPT outputs.",
    )
    parser.add_argument(
        "--include-systems",
        default="",
        help="Comma-separated system IDs. Empty means all systems.",
    )
    parser.add_argument(
        "--include-cases",
        default="",
        help="Comma-separated case IDs. Empty means all cases.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Locked text model for benchmark runs.",
    )
    parser.add_argument(
        "--cases-root",
        default=str(BENCHMARK_DIR / "tknt_case_artifacts"),
        help="Where TKNT pipeline case artifacts are written.",
    )
    parser.add_argument("--user", default="benchmark_user")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument(
        "--external-timeout-s",
        type=int,
        default=None,
        help=(
            "Optional timeout for IDesign/FlairGPT subprocesses. Use 0 to disable it."
        ),
    )
    parser.add_argument(
        "--run-external",
        action="store_true",
        help="Run IDesign/FlairGPT automatically instead of only reusing external outputs.",
    )
    parser.add_argument(
        "--idesign-objects",
        type=int,
        default=8,
        help="Fixed object budget passed to IDesign for each benchmark case.",
    )
    parser.add_argument(
        "--external-max-attempts",
        type=int,
        default=1,
        help="Maximum attempts for automatically run external systems.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = _select_cases(
        cases=_load_cases(Path(args.cases)),
        include_cases=args.include_cases,
    )
    systems = _select_systems(
        systems=_load_systems(Path(args.systems)),
        include_systems=args.include_systems,
    )
    plan_rows = _build_plan(
        cases=cases,
        systems=systems,
        output_root=output_root,
        model=args.model,
    )
    _write_jsonl(output_root / "run_plan.jsonl", plan_rows)

    if args.mode == "plan":
        _write_pending_external_jobs(
            plan_rows=plan_rows,
            external_root=Path(args.external_root),
            output_root=output_root,
        )
        print(f"Wrote benchmark plan: {output_root / 'run_plan.jsonl'}")
        return 0

    if args.mode == "run":
        _run_plan(
            plan_rows=plan_rows,
            cases=cases,
            systems=systems,
            output_root=output_root,
            external_root=Path(args.external_root),
            cases_root=Path(args.cases_root),
            user=args.user,
            timeout_s=args.timeout_s,
            external_timeout_s=_external_timeout_s(
                args.external_timeout_s,
                args.timeout_s,
            ),
            max_runs=max(0, args.max_runs),
            run_external=args.run_external,
            model=args.model,
            idesign_objects=max(1, args.idesign_objects),
            external_max_attempts=max(1, args.external_max_attempts),
        )

    score_rows = _score_plan(
        plan_rows=plan_rows,
        cases=cases,
        systems=systems,
        output_root=output_root,
        external_root=Path(args.external_root),
        max_runs=max(0, args.max_runs),
    )
    _write_jsonl(output_root / "scores.jsonl", score_rows)
    _write_summary(output_root, score_rows)
    print(f"Wrote benchmark scores: {output_root / 'scores.jsonl'}")
    return 0


def _load_cases(path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = _sequence(payload)
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return [_mapping(row) for row in cases if _mapping(row)]


def _load_systems(path: Path) -> list[SystemSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    systems: list[SystemSpec] = []
    for row in _sequence(payload):
        item = _mapping(row)
        if not item:
            continue
        system_id = _string(item.get("system_id"))
        if not system_id:
            continue
        systems.append(
            SystemSpec(
                system_id=system_id,
                kind=_string(item.get("kind")) or "external",
                normalizer=_string(item.get("normalizer")) or "tknt",
                ablation=_string(item.get("ablation")) or None,
            )
        )
    if not systems:
        raise ValueError(f"No benchmark systems found in {path}")
    return systems


def _select_systems(
    *,
    systems: Sequence[SystemSpec],
    include_systems: str,
) -> list[SystemSpec]:
    requested = {
        _canonical_system_id(value.strip())
        for value in include_systems.split(",")
        if value.strip()
    }
    if not requested:
        return list(systems)
    selected = [system for system in systems if system.system_id in requested]
    missing = requested - {system.system_id for system in selected}
    if missing:
        raise ValueError(f"Unknown requested systems: {', '.join(sorted(missing))}")
    return selected


def _canonical_system_id(system_id: str) -> str:
    return SYSTEM_ID_ALIASES.get(system_id, system_id)


def _select_cases(
    *,
    cases: Sequence[Mapping[str, object]],
    include_cases: str,
) -> list[Mapping[str, object]]:
    requested = {value.strip() for value in include_cases.split(",") if value.strip()}
    if not requested:
        return list(cases)
    selected = [
        benchmark_case
        for benchmark_case in cases
        if _case_id(benchmark_case) in requested
    ]
    missing = requested - {_case_id(benchmark_case) for benchmark_case in selected}
    if missing:
        raise ValueError(f"Unknown requested cases: {', '.join(sorted(missing))}")
    return selected


def _build_plan(
    *,
    cases: Sequence[Mapping[str, object]],
    systems: Sequence[SystemSpec],
    output_root: Path,
    model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not systems:
        return rows
    for case_index, benchmark_case in enumerate(cases):
        case_id = _case_id(benchmark_case)
        rotated_systems = list(systems[case_index % len(systems) :]) + list(
            systems[: case_index % len(systems)]
        )
        for system in rotated_systems:
            run_id = f"{case_id}__{system.system_id}"
            run_dir = output_root / "runs" / run_id
            rows.append(
                {
                    "run_id": run_id,
                    "case_id": case_id,
                    "system_id": system.system_id,
                    "kind": system.kind,
                    "normalizer": system.normalizer,
                    "ablation": system.ablation,
                    "model": model,
                    "input_path": str(run_dir / "input.json"),
                    "output_path": str(run_dir / "output.json"),
                    "normalized_path": str(run_dir / "normalized.json"),
                    "normalized_variants_path": str(
                        run_dir / "normalized_variants.json"
                    ),
                    "score_path": str(run_dir / "score.json"),
                    "variant_scores_path": str(run_dir / "variant_scores.json"),
                    "stdout_path": str(run_dir / "stdout.log"),
                    "stderr_path": str(run_dir / "stderr.log"),
                }
            )
    return rows


def _run_plan(
    *,
    plan_rows: Sequence[Mapping[str, object]],
    cases: Sequence[Mapping[str, object]],
    systems: Sequence[SystemSpec],
    output_root: Path,
    external_root: Path,
    cases_root: Path,
    user: str,
    timeout_s: int,
    external_timeout_s: int | None,
    max_runs: int,
    run_external: bool,
    model: str,
    idesign_objects: int,
    external_max_attempts: int,
) -> None:
    _ = output_root
    system_by_id = {system.system_id: system for system in systems}
    case_by_id = {_case_id(benchmark_case): benchmark_case for benchmark_case in cases}
    pending_external: list[dict[str, object]] = []
    for index, row in enumerate(plan_rows):
        if max_runs and index >= max_runs:
            break
        system = system_by_id[_string(row.get("system_id"))]
        benchmark_case = case_by_id[_string(row.get("case_id"))]
        output_path = Path(_string(row.get("output_path")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if system.kind == "tknt" or run_external:
            _clear_stale_run_artifacts(row)
        if system.kind == "tknt":
            _run_tknt_case(
                row=row,
                benchmark_case=benchmark_case,
                system=system,
                cases_root=cases_root,
                user=user,
                timeout_s=timeout_s,
            )
        else:
            if run_external:
                _run_external_case(
                    row=row,
                    benchmark_case=benchmark_case,
                    system=system,
                    model=model,
                    timeout_s=external_timeout_s,
                    idesign_objects=idesign_objects,
                    max_attempts=external_max_attempts,
                )
                continue
            source_path = _find_external_output(
                external_root=external_root,
                case_id=_case_id(benchmark_case),
                system_id=system.system_id,
            )
            if source_path is None:
                pending_external.append(
                    {
                        "case_id": _case_id(benchmark_case),
                        "system_id": system.system_id,
                        "expected_paths": _external_output_candidates(
                            external_root=external_root,
                            case_id=_case_id(benchmark_case),
                            system_id=system.system_id,
                        ),
                    }
                )
                continue
            payload = load_output_payload(source_path)
            _write_json(output_path, payload)
    _write_jsonl(output_root / "pending_external_jobs.jsonl", pending_external)


def _run_external_case(
    *,
    row: Mapping[str, object],
    benchmark_case: Mapping[str, object],
    system: SystemSpec,
    model: str,
    timeout_s: int | None,
    idesign_objects: int,
    max_attempts: int,
) -> None:
    run_external_case(
        row=row,
        benchmark_case=benchmark_case,
        system_id=system.system_id,
        model=model,
        timeout_s=timeout_s,
        idesign_objects=idesign_objects,
        max_attempts=max_attempts,
    )


def _external_timeout_s(value: int | None, fallback: int) -> int | None:
    timeout_s = fallback if value is None else value
    if timeout_s <= 0:
        return None
    return timeout_s


def _run_tknt_case(
    *,
    row: Mapping[str, object],
    benchmark_case: Mapping[str, object],
    system: SystemSpec,
    cases_root: Path,
    user: str,
    timeout_s: int,
) -> None:
    input_path = Path(_string(row.get("input_path")))
    output_path = Path(_string(row.get("output_path")))
    stdout_path = Path(_string(row.get("stdout_path")))
    stderr_path = Path(_string(row.get("stderr_path")))
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_payload = _mapping(benchmark_case.get("input_payload"))
    if not input_payload:
        raise ValueError(f"Case {_case_id(benchmark_case)} has no input_payload.")
    _write_json(input_path, input_payload)

    env = os.environ.copy()
    locked_model = _string(row.get("model")) or DEFAULT_MODEL
    env["TKNT_DISABLE_MODEL_FALLBACK"] = env.get(
        "TKNT_DISABLE_MODEL_FALLBACK",
        "1",
    )
    env["TKNT_STYLE_POLICY_REQUIRE_LLM"] = "1"
    env["TKNT_BENCHMARK_TEXT_MODEL"] = locked_model
    if system.ablation:
        env["TKNT_ABLATION_MODE"] = system.ablation
    _configure_writable_cache_env(env, output_path.parent)

    command = [
        env.get("TKNT_PYTHON") or sys.executable,
        str(REPO_ROOT / "run_case_backend_cli.py"),
        "--input",
        str(input_path),
        "--user",
        user,
        "--cases-root",
        str(cases_root),
        "--case-id",
        _string(row.get("run_id")),
        "--stdout",
        "wrapped",
        "--out",
        str(output_path),
        "--ablation",
        system.ablation or "full",
    ]
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            if _truthy_env(WRITE_STDOUT_LOG_ENV):
                stdout_target = stack.enter_context(
                    stdout_path.open("w", encoding="utf-8")
                )
            else:
                _unlink_if_exists(stdout_path)
                stdout_target = subprocess.DEVNULL
            stderr_file = stack.enter_context(stderr_path.open("w", encoding="utf-8"))
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=stdout_target,
                stderr=stderr_file,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        elapsed_s = time.perf_counter() - started
        _write_json(
            output_path,
            {
                "error": "tknt_run_timeout",
                "exit_code": None,
                "runtime_s": elapsed_s,
                "stderr_preview": _read_text_tail(stderr_path, max_chars=4000),
                "final_output": None,
            },
        )
        return
    elapsed_s = time.perf_counter() - started

    if output_path.exists():
        payload = load_output_payload(output_path)
        if isinstance(payload, Mapping):
            wrapped_payload = dict(payload)
            wrapped_payload["runtime_s"] = elapsed_s
            wrapped_payload["exit_code"] = completed.returncode
            _mirror_tknt_module_artifacts(
                payload=wrapped_payload,
                run_dir=output_path.parent,
                benchmark_case=benchmark_case,
            )
            _write_json(output_path, wrapped_payload)
        return

    _write_json(
        output_path,
        {
            "error": "tknt_run_failed",
            "exit_code": completed.returncode,
            "runtime_s": elapsed_s,
            "stderr_preview": _read_text_tail(stderr_path, max_chars=4000),
            "final_output": None,
        },
    )


def _score_plan(
    *,
    plan_rows: Sequence[Mapping[str, object]],
    cases: Sequence[Mapping[str, object]],
    systems: Sequence[SystemSpec],
    output_root: Path,
    external_root: Path,
    max_runs: int,
) -> list[dict[str, object]]:
    _ = output_root
    system_by_id = {system.system_id: system for system in systems}
    case_by_id = {_case_id(benchmark_case): benchmark_case for benchmark_case in cases}
    scores: list[dict[str, object]] = []
    for index, row in enumerate(plan_rows):
        if max_runs and index >= max_runs:
            break
        system = system_by_id[_string(row.get("system_id"))]
        benchmark_case = case_by_id[_string(row.get("case_id"))]
        output_path = Path(_string(row.get("output_path")))
        if not output_path.exists() and system.kind != "tknt":
            external_path = _find_external_output(
                external_root=external_root,
                case_id=_case_id(benchmark_case),
                system_id=system.system_id,
            )
            if external_path is not None:
                output_path = external_path
        if not output_path.exists():
            scores.append(
                {
                    "case_id": _case_id(benchmark_case),
                    "system_id": system.system_id,
                    "overall_score": 0.0,
                    "error": "missing_output",
                }
            )
            continue
        try:
            payload = load_output_payload(output_path)
            failed_score = _failed_output_score(
                payload=payload,
                benchmark_case=benchmark_case,
                system_id=system.system_id,
            )
            if failed_score is not None:
                score_path = Path(_string(row.get("score_path")))
                _write_json(score_path, failed_score)
                scores.append(failed_score)
                continue
            variant_score = _score_tknt_final_variants(
                row=row,
                payload=payload,
                benchmark_case=benchmark_case,
                system=system,
                output_path=output_path,
            )
            if variant_score is not None:
                score_path = Path(_string(row.get("score_path")))
                _write_json(score_path, variant_score)
                scores.append(variant_score)
                continue
            layout = normalize_payload(
                normalizer=system.normalizer,
                payload=payload,
                benchmark_case=benchmark_case,
                system_id=system.system_id,
                source_path=output_path,
            )
            normalized_path = Path(_string(row.get("normalized_path")))
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(normalized_path, normalized_layout_to_json(layout))
            score = score_layout(benchmark_case=benchmark_case, layout=layout)
            _promote_score_metric_fields(score)
            _attach_llm_retry_report(score, payload)
            score_path = Path(_string(row.get("score_path")))
            _write_json(score_path, score)
            scores.append(score)
        except Exception as exc:  # pragma: no cover - benchmark fault capture
            scores.append(
                {
                    "case_id": _case_id(benchmark_case),
                    "system_id": system.system_id,
                    "overall_score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return scores


def _score_tknt_final_variants(
    *,
    row: Mapping[str, object],
    payload: object,
    benchmark_case: Mapping[str, object],
    system: SystemSpec,
    output_path: Path,
) -> dict[str, object] | None:
    if system.normalizer != "tknt":
        return None
    variants_payload = _load_tknt_layout_variants(payload)
    if variants_payload is None:
        return None
    variants = [
        _mapping(variant)
        for variant in _sequence(variants_payload.get("variants"))
        if _mapping(variant)
    ]
    if not variants:
        return None

    root = _mapping(payload)
    variant_scores: list[dict[str, object]] = []
    normalized_variants: list[dict[str, object]] = []
    for index, variant in enumerate(variants, start=1):
        styled_result = _mapping(variant.get("styled_result"))
        if not styled_result:
            continue
        variant_id = _string(variant.get("variant_id")) or f"variant_{index}"
        variant_payload = dict(root)
        variant_payload["final_output"] = styled_result
        layout = normalize_payload(
            normalizer=system.normalizer,
            payload=variant_payload,
            benchmark_case=benchmark_case,
            system_id=system.system_id,
            source_path=output_path,
        )
        score = score_layout(benchmark_case=benchmark_case, layout=layout)
        _promote_score_metric_fields(score)
        _attach_llm_retry_report(score, payload)
        score.update(
            {
                "variant_id": variant_id,
                "variant_index": index,
                "variant_label": _string(variant.get("label")) or f"Option {index}",
                "variant_source": _string(variant.get("source")),
                "variant_reason": _string(variant.get("reason")),
                "gallery_eligible": bool(variant.get("gallery_eligible", False)),
                "complete": bool(variant.get("complete", False)),
                "coverage_ratio": _float(variant.get("coverage_ratio")),
                "missing_cluster_ids": list(
                    _sequence(variant.get("missing_cluster_ids"))
                ),
                "original_layout_score": _float(variant.get("layout_score")),
            }
        )
        variant_scores.append(score)
        normalized_layout = normalized_layout_to_json(layout)
        normalized_layout.update(
            {
                "variant_id": variant_id,
                "variant_index": index,
                "variant_label": score["variant_label"],
                "variant_source": score["variant_source"],
            }
        )
        normalized_variants.append(normalized_layout)

    if not variant_scores:
        return None

    selected_index, selected_score = max(
        enumerate(variant_scores),
        key=lambda item: (
            _float(item[1].get("overall_score")),
            _float(_mapping(item[1].get("component_scores")).get("geometry")),
            _float(_mapping(item[1].get("component_scores")).get("request_following")),
            -int(item[1].get("variant_index") or 0),
        ),
    )
    canonical_score = variant_scores[0]
    selected_variant_id = _string(selected_score.get("variant_id"))
    canonical_variant_id = _string(canonical_score.get("variant_id"))

    variant_scores_path = Path(_string(row.get("variant_scores_path")))
    normalized_variants_path = Path(_string(row.get("normalized_variants_path")))
    normalized_path = Path(_string(row.get("normalized_path")))
    _write_json(
        variant_scores_path,
        {
            "case_id": _case_id(benchmark_case),
            "system_id": system.system_id,
            "selection_policy": "best_overall_score_across_final_gallery",
            "selected_variant_id": selected_variant_id,
            "canonical_variant_id": canonical_variant_id,
            "scores": variant_scores,
        },
    )
    _write_json(
        normalized_variants_path,
        {
            "case_id": _case_id(benchmark_case),
            "system_id": system.system_id,
            "selected_variant_id": selected_variant_id,
            "canonical_variant_id": canonical_variant_id,
            "variants": normalized_variants,
        },
    )
    _write_json(normalized_path, normalized_variants[selected_index])

    out = dict(selected_score)
    out["variant_evaluation"] = {
        "selection_policy": "best_overall_score_across_final_gallery",
        "variant_count": len(variant_scores),
        "selected_variant_id": selected_variant_id,
        "selected_variant_index": selected_score.get("variant_index"),
        "canonical_variant_id": canonical_variant_id,
        "canonical_overall_score": canonical_score.get("overall_score"),
        "best_overall_score": selected_score.get("overall_score"),
        "variant_scores_path": str(variant_scores_path),
        "normalized_variants_path": str(normalized_variants_path),
    }
    out["canonical_score"] = canonical_score
    return out


def _failed_output_score(
    *,
    payload: object,
    benchmark_case: Mapping[str, object],
    system_id: str,
) -> dict[str, object] | None:
    root = _mapping(payload)
    if not root:
        return None
    exit_code = root.get("exit_code")
    has_error = bool(root.get("error"))
    failed_exit = exit_code is not None and _float(exit_code) != 0.0
    if not has_error and not failed_exit:
        return None

    error = _string(root.get("error")) or f"exit_code_{_string(exit_code)}"
    score = {
        "case_id": _case_id(benchmark_case),
        "system_id": system_id,
        "overall_score": 0.0,
        "error": error,
        "runtime_s": _float(root.get("runtime_s")),
        "input_tokens": root.get("input_tokens"),
        "output_tokens": root.get("output_tokens"),
        "metrics": {
            "runtime_s": _float(root.get("runtime_s")),
            "input_tokens": root.get("input_tokens"),
            "output_tokens": root.get("output_tokens"),
            "objects_total": 0,
            "blocking_objects_total": 0,
            "valid_blocking_object_count": 0,
            "objects_out_of_bounds": 0,
            "overlapping_blocking_pairs": 0,
        },
    }
    _attach_llm_retry_report(score, root)
    return score


def _attach_llm_retry_report(
    score: dict[str, object],
    payload: object,
) -> None:
    root = _mapping(payload)
    usage = _mapping(root.get("llm_usage"))
    retry_events = usage.get("retry_events")
    retry_summary = usage.get("retry_summary")
    if isinstance(retry_events, list):
        score["llm_retry_events"] = retry_events
    if isinstance(retry_summary, Mapping):
        score["llm_retry_summary"] = dict(retry_summary)


def _promote_score_metric_fields(score: dict[str, object]) -> None:
    metrics = _mapping(score.get("metrics"))
    for key in ("runtime_s", "input_tokens", "output_tokens"):
        if key in metrics:
            score[key] = metrics.get(key)


def _load_tknt_layout_variants(payload: object) -> Mapping[str, object] | None:
    root = _mapping(payload)
    embedded = _mapping(root.get("layout_variants"))
    if embedded:
        return embedded

    case_dir = _string(root.get("case_dir"))
    if not case_dir:
        return None
    case_path = Path(case_dir)
    candidates = (
        case_path / "module_outputs" / "layout_variants.json",
        case_path / "09_layout_variants.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        payload_from_file = load_output_payload(path)
        variants_payload = _mapping(payload_from_file)
        if variants_payload:
            return variants_payload
    return None


def _mirror_tknt_module_artifacts(
    *,
    payload: dict[str, object],
    run_dir: Path,
    benchmark_case: Mapping[str, object] | None = None,
) -> None:
    case_dir = _string(payload.get("case_dir"))
    if not case_dir:
        return

    case_path = Path(case_dir)
    source_dir = case_path / "module_outputs"
    if not source_dir.is_dir():
        return

    destination_dir = run_dir / "module_outputs"
    _replace_directory(source=source_dir, destination=destination_dir)
    payload["case_module_outputs_path"] = str(source_dir)
    payload["run_module_outputs_path"] = str(destination_dir)

    manifest_path = case_path / "module_io_manifest.json"
    if manifest_path.exists():
        shutil.copy2(manifest_path, run_dir / "module_io_manifest.json")

    _write_drop_diagnostics(
        source_dir=source_dir,
        destination_path=run_dir / "drop_diagnostics.json",
    )
    _write_request_awareness_diagnostics(
        source_dir=source_dir,
        destination_path=run_dir / "request_awareness_diagnostics.json",
        input_path=run_dir / "input.json",
        benchmark_case=benchmark_case,
    )


def _replace_directory(*, source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copytree(source, destination)


def _write_request_awareness_diagnostics(
    *,
    source_dir: Path,
    destination_path: Path,
    input_path: Path,
    benchmark_case: Mapping[str, object] | None,
) -> None:
    cluster_forge = _load_json_mapping(source_dir / "cluster_forge.json")
    cluster_merger = _load_json_mapping(source_dir / "cluster_output_merger.json")
    tier_count = _load_json_mapping(source_dir / "tier_count_director.json")

    semantic_program = _mapping(cluster_forge.get("semantic_layout_program"))
    if not semantic_program:
        semantic_program = _mapping(cluster_merger.get("semantic_layout_program"))
    if not semantic_program:
        return

    active_clusters = [
        _request_cluster_summary(_mapping(cluster))
        for cluster in _sequence(semantic_program.get("active_clusters"))
        if _mapping(cluster)
    ]
    tier_count_decisions = [
        _tier_count_decision_summary(_mapping(row))
        for row in _sequence(tier_count.get("decisions"))
        if _mapping(row)
    ]
    diagnostics = {
        "input": _load_input_brief(input_path),
        "benchmark_expected": _mapping(
            _mapping(benchmark_case).get("expected") if benchmark_case else None
        ),
        "source_files": {
            "cluster_forge": str(source_dir / "cluster_forge.json"),
            "cluster_output_merger": str(source_dir / "cluster_output_merger.json"),
            "tier_count_director": str(source_dir / "tier_count_director.json"),
        },
        "semantic_layout_program": {
            "selection_constraints": _mapping(
                semantic_program.get("selection_constraints")
            ),
            "controlled_degradation": _mapping(
                semantic_program.get("controlled_degradation")
            ),
            "notes": list(_sequence(semantic_program.get("notes"))),
            "active_clusters": active_clusters,
        },
        "tier_count_handoff": {
            "budget_mode": _string(tier_count.get("budget_mode")),
            "budget_valid": tier_count.get("budget_valid"),
            "decisions": tier_count_decisions,
        },
        "request_signal_issues": _request_signal_issues(
            active_clusters,
            benchmark_case=benchmark_case,
        ),
    }
    _write_json(destination_path, diagnostics)


def _request_cluster_summary(cluster: Mapping[str, object]) -> dict[str, object]:
    hints = _mapping(cluster.get("tier_count_hints"))
    hint_by_type = {
        _string(hint.get("object_type")): _mapping(hint)
        for hint in _sequence(hints.get("object_hints"))
        if _mapping(hint) and _string(_mapping(hint).get("object_type"))
    }
    bundles = [
        _request_bundle_summary(_mapping(bundle), hint_by_type=hint_by_type)
        for bundle in _sequence(cluster.get("required_bundles"))
        if _mapping(bundle)
    ]
    return {
        "cluster_id": _string(cluster.get("cluster_id")),
        "layout_role": _string(cluster.get("layout_role")),
        "priority": _string(cluster.get("priority")),
        "activation_reason": _string(cluster.get("activation_reason")),
        "viability_score": _mapping(cluster.get("viability_score")),
        "tier_count_hints": {
            "bundle_class": _string(hints.get("bundle_class")),
            "preserve_level": _string(hints.get("preserve_level")),
            "keep_if_space_surplus": hints.get("keep_if_space_surplus"),
            "space_surplus_threshold": hints.get("space_surplus_threshold"),
            "drop_order_bias": _string(hints.get("drop_order_bias")),
        },
        "required_bundles": bundles,
    }


def _request_bundle_summary(
    bundle: Mapping[str, object],
    *,
    hint_by_type: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    objects = []
    for obj in _sequence(bundle.get("objects")):
        obj_map = _mapping(obj)
        object_type = _string(obj_map.get("object_type"))
        hint = hint_by_type.get(object_type, {})
        objects.append(
            {
                "object_type": object_type,
                "role": _string(obj_map.get("role")),
                "required": obj_map.get("required"),
                "max_keep": obj_map.get("max_keep"),
                "tier_min_keep": hint.get("min_keep"),
                "tier_max_keep": hint.get("max_keep"),
                "tier_keep_if_space_surplus": hint.get("keep_if_space_surplus"),
                "tier_space_surplus_threshold": hint.get("space_surplus_threshold"),
                "tier_drop_order_bias": _string(hint.get("drop_order_bias")),
                "tier_preserve_level": _string(hint.get("preserve_level")),
                "tier_preferred_size": hint.get("preferred_size_tier"),
            }
        )
    return {
        "bundle_id": _string(bundle.get("bundle_id")),
        "objects": objects,
    }


def _tier_count_decision_summary(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "cluster_id": _string(row.get("cluster_id")),
        "object_type": _string(row.get("object_type") or row.get("category")),
        "quantity": _int(row.get("quantity")),
        "size_tier": row.get("size_tier"),
        "priority": _string(row.get("priority")),
        "preserve_level": _string(row.get("preserve_level")),
        "role": _string(row.get("role")),
        "utility_score": row.get("utility_score"),
    }


def _request_signal_issues(
    active_clusters: Sequence[Mapping[str, object]],
    *,
    benchmark_case: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    semantic_objects = _semantic_objects_by_canonical_type(active_clusters)
    for cluster in active_clusters:
        cluster_id = _string(cluster.get("cluster_id"))
        layout_role = _string(cluster.get("layout_role"))
        viability = _mapping(cluster.get("viability_score"))
        brief_support = _float(viability.get("brief_support"))
        for bundle in _sequence(cluster.get("required_bundles")):
            bundle_map = _mapping(bundle)
            for obj in _sequence(bundle_map.get("objects")):
                obj_map = _mapping(obj)
                object_type = _string(obj_map.get("object_type"))
                required = obj_map.get("required") is True
                min_keep = _int(obj_map.get("tier_min_keep"))
                if required and min_keep <= 0:
                    issues.append(
                        {
                            "kind": "required_object_has_zero_tier_min_keep",
                            "cluster_id": cluster_id,
                            "object_type": object_type,
                            "brief_support": brief_support,
                            "impact": "Tier Count may legally drop this object.",
                        }
                    )
                if required and layout_role == "optional":
                    issues.append(
                        {
                            "kind": "required_object_inside_optional_cluster",
                            "cluster_id": cluster_id,
                            "object_type": object_type,
                            "brief_support": brief_support,
                            "impact": "Budget repair may remove the whole cluster early.",
                        }
                    )
    issues.extend(
        _benchmark_request_signal_issues(
            semantic_objects=semantic_objects,
            benchmark_case=benchmark_case,
        )
    )
    return issues


def _semantic_objects_by_canonical_type(
    active_clusters: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    objects_by_type: dict[str, list[Mapping[str, object]]] = {}
    for cluster in active_clusters:
        cluster_id = _string(cluster.get("cluster_id"))
        layout_role = _string(cluster.get("layout_role"))
        viability = _mapping(cluster.get("viability_score"))
        for bundle in _sequence(cluster.get("required_bundles")):
            bundle_map = _mapping(bundle)
            for obj in _sequence(bundle_map.get("objects")):
                obj_map = _mapping(obj)
                object_type = _string(obj_map.get("object_type"))
                if not object_type:
                    continue
                row = dict(obj_map)
                row["cluster_id"] = cluster_id
                row["layout_role"] = layout_role
                row["brief_support"] = viability.get("brief_support")
                objects_by_type.setdefault(
                    canonical_object_type(object_type), []
                ).append(row)
    return objects_by_type


def _benchmark_request_signal_issues(
    *,
    semantic_objects: Mapping[str, Sequence[Mapping[str, object]]],
    benchmark_case: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if benchmark_case is None:
        return []

    expected = _mapping(benchmark_case.get("expected"))
    issues: list[dict[str, object]] = []
    for object_type in _sequence(expected.get("required_objects")):
        canonical = canonical_object_type(_string(object_type))
        semantic_rows = semantic_objects.get(canonical, [])
        if not semantic_rows:
            issues.append(
                {
                    "kind": "benchmark_required_object_missing_from_semantic_program",
                    "object_type": canonical,
                    "impact": "Upstream request awareness did not carry this expected object forward.",
                }
            )
            continue
        if not any(row.get("required") is True for row in semantic_rows):
            issues.append(
                {
                    "kind": "benchmark_required_object_not_marked_required",
                    "object_type": canonical,
                    "semantic_rows": list(semantic_rows),
                    "impact": "Tier Count may treat this expected object as surplus support.",
                }
            )

    count_ranges = _mapping(expected.get("count_ranges"))
    for object_type, raw_range in count_ranges.items():
        range_map = _mapping(raw_range)
        minimum = _int(range_map.get("min"))
        if minimum <= 0:
            continue
        canonical = canonical_object_type(_string(object_type))
        semantic_rows = semantic_objects.get(canonical, [])
        max_min_keep = max(
            [_int(row.get("tier_min_keep")) for row in semantic_rows],
            default=0,
        )
        if max_min_keep < minimum:
            issues.append(
                {
                    "kind": "benchmark_min_count_exceeds_tier_min_keep",
                    "object_type": canonical,
                    "expected_min": minimum,
                    "semantic_tier_min_keep": max_min_keep,
                    "semantic_rows": list(semantic_rows),
                    "impact": "Tier Count can satisfy its own hints while failing benchmark minimum count.",
                }
            )
    return issues


def _load_input_brief(path: Path) -> dict[str, object]:
    payload = _load_json_mapping(path)
    user_input = _mapping(payload.get("user_input"))
    return {
        "description": _string(user_input.get("description")),
        "room_type": _string(user_input.get("room_type")),
        "style": _string(user_input.get("style")),
        "floor_area_m2": user_input.get("floor_area_m2"),
    }


def _write_drop_diagnostics(*, source_dir: Path, destination_path: Path) -> None:
    tier_count = _load_json_mapping(source_dir / "tier_count_director.json")
    cluster_merger = _load_json_mapping(source_dir / "cluster_output_merger.json")
    solver_drops = _load_json_mapping(
        source_dir / "macro_cluster_solver_dropped_inventory.json"
    )

    if not tier_count and not cluster_merger and not solver_drops:
        return

    tier_decisions = [
        _mapping(row) for row in _sequence(tier_count.get("decisions")) if _mapping(row)
    ]
    cluster_decisions = [
        _mapping(row)
        for row in _sequence(tier_count.get("cluster_decisions"))
        if _mapping(row)
    ]
    decision_reasons = _tier_decision_reasons_by_key(cluster_decisions)
    diagnostics = {
        "tier_count": {
            "budget_mode": _string(tier_count.get("budget_mode")),
            "budget_valid": tier_count.get("budget_valid"),
            "decision_summary": _mapping(tier_count.get("decision_summary")),
            "budget_adjustments": list(_sequence(tier_count.get("budget_adjustments"))),
            "dropped_decisions": [
                _drop_decision_summary(row, decision_reasons=decision_reasons)
                for row in tier_decisions
                if _int(row.get("quantity")) <= 0
            ],
            "kept_decisions": [
                _drop_decision_summary(row, decision_reasons=decision_reasons)
                for row in tier_decisions
                if _int(row.get("quantity")) > 0
            ],
            "dropped_cluster_ids": [
                _string(row.get("cluster_id"))
                for row in cluster_decisions
                if _string(row.get("decision_status")) == "dropped"
            ],
        },
        "cluster_output_merger": {
            "active_cluster_ids": list(
                _sequence(cluster_merger.get("active_cluster_ids"))
            ),
        },
        "macro_cluster_solver": {
            "dropped_inventory_by_cluster": _mapping(
                solver_drops.get("dropped_inventory_by_cluster")
            ),
            "total_dropped_items": solver_drops.get("total_dropped_items"),
            "clusters": list(_sequence(solver_drops.get("clusters"))),
        },
    }
    _write_json(destination_path, diagnostics)


def _tier_decision_reasons_by_key(
    cluster_decisions: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], str]:
    reasons: dict[tuple[str, str], str] = {}
    for cluster in cluster_decisions:
        cluster_id = _string(cluster.get("cluster_id"))
        for bundle in _sequence(cluster.get("selected_bundles")):
            bundle_map = _mapping(bundle)
            if not bundle_map:
                continue
            for obj in _sequence(bundle_map.get("objects")):
                object_map = _mapping(obj)
                object_type = _string(object_map.get("object_type"))
                reason = _string(object_map.get("decision_reason"))
                if cluster_id and object_type and reason:
                    reasons[(cluster_id, object_type)] = reason
    return reasons


def _drop_decision_summary(
    row: Mapping[str, object],
    *,
    decision_reasons: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    cluster_id = _string(row.get("cluster_id"))
    object_type = _string(row.get("object_type") or row.get("category"))
    return {
        "cluster_id": cluster_id,
        "object_type": object_type,
        "quantity": _int(row.get("quantity")),
        "size_tier": row.get("size_tier"),
        "priority": _string(row.get("priority")),
        "preserve_level": _string(row.get("preserve_level")),
        "role": _string(row.get("role")),
        "utility_score": row.get("utility_score"),
        "decision_reason": decision_reasons.get(
            (cluster_id, object_type),
            _string(row.get("rationale")),
        ),
    }


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        payload = load_output_payload(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return _mapping(payload)


def _write_summary(
    output_root: Path, score_rows: Sequence[Mapping[str, object]]
) -> None:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in score_rows:
        system_id = _string(row.get("system_id"))
        grouped.setdefault(system_id, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for system_id, rows in sorted(grouped.items()):
        scores = [_float(row.get("overall_score")) for row in rows]
        summary_rows.append(
            {
                "system_id": system_id,
                "case_count": len(rows),
                "mean_overall_score": round(sum(scores) / max(len(scores), 1), 3),
                "missing_or_failed": len([row for row in rows if row.get("error")]),
                "mean_runtime_s": _mean_metric(rows, "runtime_s"),
                "mean_input_tokens": _mean_metric(rows, "input_tokens"),
                "mean_output_tokens": _mean_metric(rows, "output_tokens"),
                "mean_canonical_overall_score": _mean_nested_metric(
                    rows, "variant_evaluation", "canonical_overall_score"
                ),
                "mean_final_variant_count": _mean_nested_metric(
                    rows, "variant_evaluation", "variant_count"
                ),
                "mean_objects_out_of_bounds": _mean_nested_metric(
                    rows, "metrics", "objects_out_of_bounds"
                ),
                "mean_overlapping_blocking_pairs": _mean_nested_metric(
                    rows, "metrics", "overlapping_blocking_pairs"
                ),
                "mean_valid_blocking_object_count": _mean_nested_metric(
                    rows, "metrics", "valid_blocking_object_count"
                ),
            }
        )
    _write_json(output_root / "summary.json", summary_rows)
    csv_path = output_root / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["system_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_pending_external_jobs(
    *,
    plan_rows: Sequence[Mapping[str, object]],
    external_root: Path,
    output_root: Path,
) -> None:
    pending: list[dict[str, object]] = []
    for row in plan_rows:
        if row.get("kind") == "tknt":
            continue
        case_id = _string(row.get("case_id"))
        system_id = _string(row.get("system_id"))
        pending.append(
            {
                "case_id": case_id,
                "system_id": system_id,
                "put_output_at_one_of": _external_output_candidates(
                    external_root=external_root,
                    case_id=case_id,
                    system_id=system_id,
                ),
            }
        )
    _write_jsonl(output_root / "pending_external_jobs.jsonl", pending)


def _find_external_output(
    *,
    external_root: Path,
    case_id: str,
    system_id: str,
) -> Path | None:
    for candidate in _external_output_candidates(
        external_root=external_root,
        case_id=case_id,
        system_id=system_id,
    ):
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _external_output_candidates(
    *,
    external_root: Path,
    case_id: str,
    system_id: str,
) -> list[str]:
    return [
        str(external_root / system_id / case_id / "output.json"),
        str(external_root / system_id / case_id / "output.txt"),
        str(external_root / case_id / system_id / "output.json"),
        str(external_root / case_id / system_id / "output.txt"),
    ]


def _mean_metric(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _mean_nested_metric(
    rows: Sequence[Mapping[str, object]],
    section: str,
    key: str,
) -> float | None:
    values: list[float] = []
    for row in rows:
        section_map = _mapping(row.get(section))
        if key in section_map:
            values.append(_float(section_map.get(key)))
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _clear_stale_run_artifacts(row: Mapping[str, object]) -> None:
    output_value = _string(row.get("output_path"))
    if output_value:
        run_dir = Path(output_value).parent
        _unlink_if_exists(run_dir / "status.json")
        _unlink_if_exists(run_dir / "module_io_manifest.json")
        _unlink_if_exists(run_dir / "drop_diagnostics.json")
        _unlink_if_exists(run_dir / "request_awareness_diagnostics.json")
        _remove_directory_if_exists(run_dir / "module_outputs")
    for key in (
        "output_path",
        "normalized_path",
        "normalized_variants_path",
        "score_path",
        "variant_scores_path",
    ):
        value = _string(row.get(key))
        if value:
            _unlink_if_exists(Path(value))


def _configure_writable_cache_env(env: dict[str, str], run_dir: Path) -> None:
    cache_dir = run_dir / ".cache"
    matplotlib_dir = cache_dir / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    env.setdefault("MPLCONFIGDIR", str(matplotlib_dir))


def _read_text_tail(path: Path, *, max_chars: int) -> str:
    if max_chars <= 0 or not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - max_chars * 4), os.SEEK_SET)
        return file.read().decode("utf-8", errors="replace")[-max_chars:]


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _remove_directory_if_exists(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _case_id(benchmark_case: Mapping[str, object]) -> str:
    value = _string(benchmark_case.get("case_id"))
    if not value:
        raise ValueError("Benchmark case is missing case_id.")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
