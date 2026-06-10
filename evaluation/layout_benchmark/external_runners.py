from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from evaluation.layout_benchmark.cache_env import configure_fresh_cache_env

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_MODEL = "gpt-5.4-nano"
WRITE_STDOUT_LOG_ENV = "LAYOUT_BENCHMARK_WRITE_STDOUT_LOG"
SYSTEM_ID_ALIASES: Mapping[str, str] = {
    "idesign_gemini": "idesign_openai",
    "flairgpt_gemini": "flairgpt_openai",
}


def run_external_case(
    *,
    row: Mapping[str, object],
    benchmark_case: Mapping[str, object],
    system_id: str,
    model: str,
    timeout_s: int | None,
    idesign_objects: int,
    max_attempts: int = 1,
) -> None:
    input_path = Path(_string(row.get("input_path")))
    output_path = Path(_string(row.get("output_path")))
    stdout_path = Path(_string(row.get("stdout_path")))
    stderr_path = Path(_string(row.get("stderr_path")))
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(input_path, _mapping(benchmark_case.get("input_payload")))
    max_attempts = max(1, max_attempts)

    command, cwd, env = _external_command(
        system_id=system_id,
        input_path=input_path.resolve(),
        output_path=output_path.resolve(),
        benchmark_case=benchmark_case,
        model=model,
        idesign_objects=idesign_objects,
    )

    attempts: list[dict[str, object]] = []
    for attempt_number in range(1, max_attempts + 1):
        _unlink_if_exists(output_path)
        attempt_env = dict(env)
        configure_fresh_cache_env(attempt_env, output_path.parent)
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
                stderr_file = stack.enter_context(
                    stderr_path.open("w", encoding="utf-8")
                )
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=attempt_env,
                    text=True,
                    stdout=stdout_target,
                    stderr=stderr_file,
                    timeout=timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            elapsed_s = time.perf_counter() - started
            attempts.append(
                _attempt_summary(
                    attempt_number=attempt_number,
                    status="failed",
                    error="external_run_timeout",
                    exit_code=None,
                    termination=f"timed_out_after_{timeout_s}s",
                    runtime_s=elapsed_s,
                    stderr_path=stderr_path,
                )
            )
        else:
            elapsed_s = time.perf_counter() - started
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    attempts.append(
                        _attempt_summary(
                            attempt_number=attempt_number,
                            status="failed",
                            error="external_output_json_decode_failed",
                            exit_code=completed.returncode,
                            termination=f"{type(exc).__name__}: {exc}",
                            runtime_s=elapsed_s,
                            stderr_path=stderr_path,
                        )
                    )
                else:
                    if isinstance(payload, dict):
                        payload_error = _string(payload.get("error"))
                        if completed.returncode != 0 or payload_error:
                            attempts.append(
                                _attempt_summary(
                                    attempt_number=attempt_number,
                                    status="failed",
                                    error=payload_error or "external_run_failed",
                                    exit_code=completed.returncode,
                                    termination=_returncode_summary(
                                        completed.returncode
                                    ),
                                    runtime_s=elapsed_s,
                                    stderr_path=stderr_path,
                                )
                            )
                        else:
                            payload["runtime_s"] = elapsed_s
                            payload["exit_code"] = completed.returncode
                            payload["model"] = _external_model_label(
                                system_id,
                                env,
                                model,
                            )
                            payload["attempt_count"] = attempt_number
                            payload["retry_count"] = attempt_number - 1
                            if attempts:
                                payload["attempts"] = [
                                    *attempts,
                                    _attempt_summary(
                                        attempt_number=attempt_number,
                                        status="succeeded",
                                        error="",
                                        exit_code=completed.returncode,
                                        termination=_returncode_summary(
                                            completed.returncode
                                        ),
                                        runtime_s=elapsed_s,
                                        stderr_path=stderr_path,
                                    ),
                                ]
                            _write_json(output_path, payload)
                            return
                    else:
                        attempts.append(
                            _attempt_summary(
                                attempt_number=attempt_number,
                                status="failed",
                                error="external_output_not_object",
                                exit_code=completed.returncode,
                                termination=type(payload).__name__,
                                runtime_s=elapsed_s,
                                stderr_path=stderr_path,
                            )
                        )
            else:
                attempts.append(
                    _attempt_summary(
                        attempt_number=attempt_number,
                        status="failed",
                        error="external_run_failed",
                        exit_code=completed.returncode,
                        termination=_returncode_summary(completed.returncode),
                        runtime_s=elapsed_s,
                        stderr_path=stderr_path,
                    )
                )

        if attempt_number < max_attempts:
            continue

        last_attempt = attempts[-1] if attempts else {}
        _write_json(
            output_path,
            {
                "error": _string(last_attempt.get("error")) or "external_run_failed",
                "system_id": system_id,
                "exit_code": last_attempt.get("exit_code"),
                "termination": _string(last_attempt.get("termination")),
                "runtime_s": last_attempt.get("runtime_s", 0.0),
                "stderr_preview": _string(last_attempt.get("stderr_preview")),
                "objects": [],
                "model": _external_model_label(system_id, env, model),
                "attempt_count": attempt_number,
                "retry_count": attempt_number - 1,
                "attempts": attempts,
            },
        )


def _attempt_summary(
    *,
    attempt_number: int,
    status: str,
    error: str,
    exit_code: int | None,
    termination: str,
    runtime_s: float,
    stderr_path: Path,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "attempt": attempt_number,
        "status": status,
        "exit_code": exit_code,
        "termination": termination,
        "runtime_s": runtime_s,
        "stderr_preview": _read_text_tail(stderr_path, max_chars=4000),
    }
    if error:
        summary["error"] = error
    return summary


def _external_command(
    *,
    system_id: str,
    input_path: Path,
    output_path: Path,
    benchmark_case: Mapping[str, object],
    model: str,
    idesign_objects: int,
) -> tuple[list[str], Path, dict[str, str]]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"

    system_id = _canonical_system_id(system_id)
    locked_model = model or DEFAULT_MODEL
    if system_id == "idesign_openai":
        env["IDESIGN_DISABLE_CACHE"] = "1"
        env["IDESIGN_USE_OPENAI"] = "true"
        env["IDESIGN_OPENAI_MODEL"] = env.get("IDESIGN_OPENAI_MODEL") or locked_model
        cwd = REPO_ROOT / "other_systems" / "IDesign"
        return (
            [
                _python_executable("IDESIGN_PYTHON", "IDESIGN_ENV_PREFIX")
                or sys.executable,
                str(cwd / "run_benchmark_case.py"),
                "--input",
                str(input_path),
                "--out",
                str(output_path),
                "--model",
                locked_model,
                "--objects",
                str(idesign_objects),
            ],
            cwd,
            env,
        )

    if system_id == "flairgpt_openai":
        env["FLAIRGPT_USE_OPENAI"] = "true"
        env["FLAIRGPT_OPENAI_MODEL"] = env.get("FLAIRGPT_OPENAI_MODEL") or locked_model
        if not env.get("OPENAI_API_KEY") and env.get("FLAIRGPT_OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = env["FLAIRGPT_OPENAI_API_KEY"]
        cwd = REPO_ROOT / "other_systems" / "FlairGPT" / "Scene_Synthesis"
        return (
            [
                _python_executable("FLAIRGPT_PYTHON", "FLAIRGPT_ENV_PREFIX")
                or sys.executable,
                str(cwd / "run_notebook_benchmark_case.py"),
                "--scene-descriptor",
                _case_description(benchmark_case),
                "--output-json",
                str(output_path),
                "--result-id",
                _case_id(benchmark_case),
            ],
            cwd,
            env,
        )

    raise ValueError(f"Unsupported external system: {system_id}")


def _external_model_label(
    system_id: str,
    env: Mapping[str, str],
    model: str,
) -> str:
    system_id = _canonical_system_id(system_id)
    if system_id == "idesign_openai":
        return env.get("IDESIGN_OPENAI_MODEL") or model or DEFAULT_MODEL
    if system_id == "flairgpt_openai":
        return env.get("FLAIRGPT_OPENAI_MODEL") or model or DEFAULT_MODEL
    return model or DEFAULT_MODEL


def _canonical_system_id(system_id: str) -> str:
    return SYSTEM_ID_ALIASES.get(system_id, system_id)


def _python_executable(python_env: str, prefix_env: str) -> str:
    explicit_python = os.getenv(python_env, "").strip()
    if explicit_python:
        return explicit_python
    env_prefix = os.getenv(prefix_env, "").strip()
    if env_prefix:
        return str(Path(env_prefix) / "bin" / "python")
    return ""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _case_description(benchmark_case: Mapping[str, object]) -> str:
    input_payload = _mapping(benchmark_case.get("input_payload"))
    user_input = _mapping(input_payload.get("user_input"))
    description_parts = [_string(user_input.get("description")).rstrip()]
    opening_descriptions = _opening_descriptions(benchmark_case)
    if opening_descriptions:
        description_parts.extend(opening_descriptions)

    required_objects = _required_objects(benchmark_case)
    if required_objects:
        description_parts.append(f"Include {_human_join(required_objects)}.")

    relation_descriptions = _relation_descriptions(benchmark_case)
    if relation_descriptions:
        description_parts.extend(relation_descriptions)

    return " ".join(part for part in description_parts if part).strip()


def _opening_descriptions(benchmark_case: Mapping[str, object]) -> list[str]:
    bounds = _room_bounds(benchmark_case)
    if bounds is None:
        return []

    room = _mapping(benchmark_case.get("room"))
    openings = _mapping(room.get("openings"))
    descriptions: list[str] = []
    descriptions.extend(
        _opening_kind_descriptions(
            openings=_opening_rows(openings, "doors"),
            kind="door",
            bounds=bounds,
        )
    )
    descriptions.extend(
        _opening_kind_descriptions(
            openings=_opening_rows(openings, "windows"),
            kind="window",
            bounds=bounds,
        )
    )
    return descriptions


def _opening_kind_descriptions(
    *,
    openings: list[Mapping[str, object]],
    kind: str,
    bounds: tuple[float, float, float, float],
) -> list[str]:
    descriptions: list[str] = []
    for index, opening in enumerate(openings):
        segment = _opening_segment(opening)
        if segment is None:
            continue
        side, start_m, end_m = _opening_side_span(segment, bounds)
        label = _opening_label(kind=kind, index=index)
        descriptions.append(
            f"{label} is on the {side} wall, spanning roughly from "
            f"{_format_meters(start_m)}m to {_format_meters(end_m)}m along that wall."
        )
    return descriptions


def _opening_rows(
    openings: Mapping[str, object], key: str
) -> list[Mapping[str, object]]:
    rows = openings.get(key)
    if not isinstance(rows, list):
        return []
    return [_mapping(row) for row in rows]


def _opening_segment(
    opening: Mapping[str, object],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    segment = opening.get("segment_mm")
    if not isinstance(segment, list) or len(segment) < 2:
        return None
    start = _point_mm(_mapping(segment[0]))
    end = _point_mm(_mapping(segment[1]))
    if start is None or end is None:
        return None
    return start, end


def _opening_side_span(
    segment: tuple[tuple[float, float], tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> tuple[str, float, float]:
    (start_x, start_y), (end_x, end_y) = segment
    min_x, max_x, min_y, max_y = bounds
    if _same_position(start_x, end_x):
        side = "left" if _same_position(start_x, min_x) else "right"
        start_mm, end_mm = sorted((start_y - min_y, end_y - min_y))
    else:
        side = "bottom" if _same_position(start_y, min_y) else "top"
        start_mm, end_mm = sorted((start_x - min_x, end_x - min_x))
    return side, start_mm / 1000.0, end_mm / 1000.0


def _opening_label(*, kind: str, index: int) -> str:
    if kind == "door":
        if index == 0:
            return "The entry door"
        return f"Door {index + 1}"
    if index == 0:
        return "One window"
    return f"Window {index + 1}"


def _room_bounds(
    benchmark_case: Mapping[str, object],
) -> tuple[float, float, float, float] | None:
    room = _mapping(benchmark_case.get("room"))
    points = room.get("polygon_mm")
    if not isinstance(points, list):
        return None
    parsed_points = [_point_mm(_mapping(point)) for point in points]
    valid_points = [point for point in parsed_points if point is not None]
    if not valid_points:
        return None
    xs = [point[0] for point in valid_points]
    ys = [point[1] for point in valid_points]
    return min(xs), max(xs), min(ys), max(ys)


def _point_mm(point: Mapping[str, object]) -> tuple[float, float] | None:
    x = _number(point.get("x"))
    y = _number(point.get("y"))
    if x is None or y is None:
        return None
    return x, y


def _required_objects(benchmark_case: Mapping[str, object]) -> list[str]:
    expected = _mapping(benchmark_case.get("expected"))
    rows = expected.get("required_objects")
    if not isinstance(rows, list):
        return []
    return [_display_name(row) for row in rows if _display_name(row)]


def _relation_descriptions(benchmark_case: Mapping[str, object]) -> list[str]:
    expected = _mapping(benchmark_case.get("expected"))
    relations = expected.get("relations")
    if not isinstance(relations, list):
        return []

    descriptions: list[str] = []
    for relation in relations:
        relation_map = _mapping(relation)
        relation_type = _string(relation_map.get("type"))
        objects = _object_names(relation_map.get("objects"))
        opening_kind = _string(relation_map.get("opening_kind"))
        if relation_type == "avoid_near_opening":
            minimum = _number(relation_map.get("min_distance_mm"))
            if objects and opening_kind and minimum is not None:
                descriptions.append(
                    f"Keep {_human_join(objects)} at least "
                    f"{_format_meters(minimum / 1000.0)}m away from the {opening_kind}."
                )
        elif relation_type == "prefer_near_opening":
            maximum = _number(relation_map.get("max_distance_mm"))
            if objects and opening_kind and maximum is not None:
                descriptions.append(
                    f"Prefer {_human_join(objects)} within "
                    f"{_format_meters(maximum / 1000.0)}m of the {opening_kind}."
                )
    return descriptions


def _object_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_display_name(item) for item in value if _display_name(item)]


def _human_join(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _display_name(value: object) -> str:
    return _string(value).replace("_", " ")


def _format_meters(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _same_position(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6


def _case_id(benchmark_case: Mapping[str, object]) -> str:
    value = _string(benchmark_case.get("case_id"))
    if not value:
        raise ValueError("Benchmark case is missing case_id.")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_text_tail(path: Path, *, max_chars: int) -> str:
    if max_chars <= 0 or not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - max_chars * 4), os.SEEK_SET)
        return file.read().decode("utf-8", errors="replace")[-max_chars:]


def _returncode_summary(returncode: int) -> str:
    if returncode >= 0:
        return f"exited_with_code_{returncode}"
    signal_number = -returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"signal_{signal_number}"
    return f"killed_by_{signal_name}"


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
