#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
ROOT_ENV_FILE="$REPO_ROOT/.env"

REQUESTED_MODEL="${MODEL:-}"
REQUESTED_MODE="${MODE:-}"
REQUESTED_TKNT_DISABLE_MODEL_FALLBACK="${TKNT_DISABLE_MODEL_FALLBACK:-}"
REQUESTED_TKNT_ENV_PREFIX="${TKNT_ENV_PREFIX:-}"
REQUESTED_RUNS_BASE="${RUNS_BASE:-}"
REQUESTED_CASES_ROOT_BASE="${CASES_ROOT_BASE:-}"
REQUESTED_CASES="${CASES:-}"
REQUESTED_SYSTEMS="${SYSTEMS:-}"
REQUESTED_INCLUDE_SYSTEMS="${INCLUDE_SYSTEMS:-}"
REQUESTED_INCLUDE_CASES="${INCLUDE_CASES:-}"
REQUESTED_REPEATS="${REPEATS:-}"
REQUESTED_TIMEOUT_S="${TIMEOUT_S:-}"
REQUESTED_CLEAN_FAILED_OUTPUTS="${CLEAN_FAILED_OUTPUTS:-}"
REQUESTED_RUN_ANALYSIS="${RUN_ANALYSIS:-}"
REQUESTED_ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-}"
REQUESTED_POSTGRES_HOST="${POSTGRES_HOST:-}"
REQUESTED_POSTGRES_PORT="${POSTGRES_PORT:-}"
REQUESTED_POSTGRES_DB="${POSTGRES_DB:-}"
REQUESTED_POSTGRES_USER="${POSTGRES_USER:-}"
REQUESTED_POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
REQUESTED_TKNT_PYTHON="${TKNT_PYTHON:-}"
REQUESTED_BENCHMARK_RUNNER_PYTHON="${BENCHMARK_RUNNER_PYTHON:-}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

[[ -n "$REQUESTED_MODEL" ]] && MODEL="$REQUESTED_MODEL"
[[ -n "$REQUESTED_MODE" ]] && MODE="$REQUESTED_MODE"
[[ -n "$REQUESTED_TKNT_DISABLE_MODEL_FALLBACK" ]] && TKNT_DISABLE_MODEL_FALLBACK="$REQUESTED_TKNT_DISABLE_MODEL_FALLBACK"
[[ -n "$REQUESTED_TKNT_ENV_PREFIX" ]] && TKNT_ENV_PREFIX="$REQUESTED_TKNT_ENV_PREFIX"
[[ -n "$REQUESTED_RUNS_BASE" ]] && RUNS_BASE="$REQUESTED_RUNS_BASE"
[[ -n "$REQUESTED_CASES_ROOT_BASE" ]] && CASES_ROOT_BASE="$REQUESTED_CASES_ROOT_BASE"
[[ -n "$REQUESTED_CASES" ]] && CASES="$REQUESTED_CASES"
[[ -n "$REQUESTED_SYSTEMS" ]] && SYSTEMS="$REQUESTED_SYSTEMS"
[[ -n "$REQUESTED_INCLUDE_SYSTEMS" ]] && INCLUDE_SYSTEMS="$REQUESTED_INCLUDE_SYSTEMS"
[[ -n "$REQUESTED_INCLUDE_CASES" ]] && INCLUDE_CASES="$REQUESTED_INCLUDE_CASES"
[[ -n "$REQUESTED_REPEATS" ]] && REPEATS="$REQUESTED_REPEATS"
[[ -n "$REQUESTED_TIMEOUT_S" ]] && TIMEOUT_S="$REQUESTED_TIMEOUT_S"
[[ -n "$REQUESTED_CLEAN_FAILED_OUTPUTS" ]] && CLEAN_FAILED_OUTPUTS="$REQUESTED_CLEAN_FAILED_OUTPUTS"
[[ -n "$REQUESTED_RUN_ANALYSIS" ]] && RUN_ANALYSIS="$REQUESTED_RUN_ANALYSIS"
[[ -n "$REQUESTED_ANALYSIS_OUTPUT_DIR" ]] && ANALYSIS_OUTPUT_DIR="$REQUESTED_ANALYSIS_OUTPUT_DIR"
[[ -n "$REQUESTED_BENCHMARK_RUNNER_PYTHON" ]] && BENCHMARK_RUNNER_PYTHON="$REQUESTED_BENCHMARK_RUNNER_PYTHON"

: "${MODEL:=gpt-5.4-mini}"
: "${MODE:=resume}"
: "${TKNT_DISABLE_MODEL_FALLBACK:=1}"
: "${TKNT_ENV_PREFIX:=.conda_envs/prj-tknt}"
: "${RUNS_BASE:=evaluation/layout_benchmark/runs/manual_stability/runs}"
: "${CASES_ROOT_BASE:=evaluation/layout_benchmark/tknt_case_artifacts/manual_stability}"
: "${CASES:=evaluation/layout_benchmark/stability_cases.json}"
: "${SYSTEMS:=evaluation/layout_benchmark/systems_stability.json}"
: "${INCLUDE_SYSTEMS:=tknt_full,tknt_ab_no_style_policy,tknt_ab_no_request_contract,tknt_ab_no_tier_capacity,tknt_ab_no_material_macro_repair,tknt_ab_no_stylist_llm,tknt_ab_no_gallery_selector}"
: "${INCLUDE_CASES:=stability_living_japandi_movie_social,stability_living_boho_hosting_dual_zone,stability_living_industrial_bay_reading,stability_bedroom_boho_guest_vanity,stability_hard_l_bedroom_industrial_storage,stability_hard_l_living_japandi_circulation}"
: "${REPEATS:=r1}"
: "${TIMEOUT_S:=1200}"
: "${CLEAN_FAILED_OUTPUTS:=0}"
: "${RUN_ANALYSIS:=1}"
: "${ANALYSIS_OUTPUT_DIR:=evaluation/layout_benchmark/runs/manual_stability}"

if [[ -z "$REQUESTED_MODE" ]]; then
  MODE="resume"
fi
if [[ -z "$REQUESTED_TKNT_ENV_PREFIX" ]]; then
  TKNT_ENV_PREFIX=".conda_envs/prj-tknt"
fi
if [[ -z "$REQUESTED_RUNS_BASE" ]]; then
  RUNS_BASE="evaluation/layout_benchmark/runs/manual_stability/runs"
fi
if [[ -z "$REQUESTED_CASES_ROOT_BASE" ]]; then
  CASES_ROOT_BASE="evaluation/layout_benchmark/tknt_case_artifacts/manual_stability"
fi
if [[ -z "$REQUESTED_CASES" ]]; then
  CASES="evaluation/layout_benchmark/stability_cases.json"
fi
if [[ -z "$REQUESTED_SYSTEMS" ]]; then
  SYSTEMS="evaluation/layout_benchmark/systems_stability.json"
fi
if [[ -z "$REQUESTED_INCLUDE_SYSTEMS" ]]; then
  INCLUDE_SYSTEMS="tknt_full,tknt_ab_no_style_policy,tknt_ab_no_request_contract,tknt_ab_no_tier_capacity,tknt_ab_no_material_macro_repair,tknt_ab_no_stylist_llm,tknt_ab_no_gallery_selector"
fi
if [[ -z "$REQUESTED_INCLUDE_CASES" ]]; then
  INCLUDE_CASES="stability_living_japandi_movie_social,stability_living_boho_hosting_dual_zone,stability_living_industrial_bay_reading,stability_bedroom_boho_guest_vanity,stability_hard_l_bedroom_industrial_storage,stability_hard_l_living_japandi_circulation"
fi
if [[ -z "$REQUESTED_REPEATS" ]]; then
  REPEATS="r1"
fi
if [[ -z "$REQUESTED_CLEAN_FAILED_OUTPUTS" ]]; then
  CLEAN_FAILED_OUTPUTS="0"
fi
if [[ -z "$REQUESTED_RUN_ANALYSIS" ]]; then
  RUN_ANALYSIS="1"
fi
if [[ -z "$REQUESTED_ANALYSIS_OUTPUT_DIR" ]]; then
  ANALYSIS_OUTPUT_DIR="evaluation/layout_benchmark/runs/manual_stability"
fi

repo_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$value"
  fi
}

TKNT_ENV_PREFIX="$(repo_path "$TKNT_ENV_PREFIX")"
: "${TKNT_PYTHON:=$TKNT_ENV_PREFIX/bin/python}"
[[ -n "$REQUESTED_TKNT_PYTHON" ]] && TKNT_PYTHON="$REQUESTED_TKNT_PYTHON"
: "${BENCHMARK_RUNNER_PYTHON:=$TKNT_PYTHON}"

if [[ ! -x "$BENCHMARK_RUNNER_PYTHON" ]]; then
  echo "Benchmark Python not found: $BENCHMARK_RUNNER_PYTHON" >&2
  exit 1
fi

load_root_postgres_env() {
  if [[ ! -f "$ROOT_ENV_FILE" || ! -x "$TKNT_PYTHON" ]]; then
    return
  fi

  local line key value
  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      POSTGRES_HOST) POSTGRES_HOST="$value" ;;
      POSTGRES_PORT) POSTGRES_PORT="$value" ;;
      POSTGRES_DB) POSTGRES_DB="$value" ;;
      POSTGRES_USER) POSTGRES_USER="$value" ;;
      POSTGRES_PASSWORD) POSTGRES_PASSWORD="$value" ;;
    esac
  done < <("$TKNT_PYTHON" - "$ROOT_ENV_FILE" <<'PY'
import sys

from dotenv import dotenv_values

values = dotenv_values(sys.argv[1])
for key in (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
):
    value = values.get(key)
    if value is not None:
        print(f"{key}={value}")
PY
  )
}

load_root_postgres_env

[[ -n "$REQUESTED_POSTGRES_HOST" ]] && POSTGRES_HOST="$REQUESTED_POSTGRES_HOST"
[[ -n "$REQUESTED_POSTGRES_PORT" ]] && POSTGRES_PORT="$REQUESTED_POSTGRES_PORT"
[[ -n "$REQUESTED_POSTGRES_DB" ]] && POSTGRES_DB="$REQUESTED_POSTGRES_DB"
[[ -n "$REQUESTED_POSTGRES_USER" ]] && POSTGRES_USER="$REQUESTED_POSTGRES_USER"
[[ -n "$REQUESTED_POSTGRES_PASSWORD" ]] && POSTGRES_PASSWORD="$REQUESTED_POSTGRES_PASSWORD"

export TKNT_DISABLE_MODEL_FALLBACK
export TKNT_BENCHMARK_TEXT_MODEL="$MODEL"
export TKNT_PYTHON
[[ ${POSTGRES_HOST+x} ]] && export POSTGRES_HOST
[[ ${POSTGRES_PORT+x} ]] && export POSTGRES_PORT
[[ ${POSTGRES_DB+x} ]] && export POSTGRES_DB
[[ ${POSTGRES_USER+x} ]] && export POSTGRES_USER
[[ ${POSTGRES_PASSWORD+x} ]] && export POSTGRES_PASSWORD

cd "$REPO_ROOT"

clean_failed_outputs() {
  local run_root="$1"
  if [[ "$CLEAN_FAILED_OUTPUTS" != "1" ]]; then
    return
  fi
  "$BENCHMARK_RUNNER_PYTHON" - "$run_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
deleted = 0
for path in root.rglob("output.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        path.unlink()
        deleted += 1
        continue
    failed = (
        bool(payload.get("error"))
        or payload.get("exit_code") not in (None, 0)
        or payload.get("final_output") is None
    )
    if failed:
        path.unlink()
        deleted += 1
print(f"Deleted failed output.json files under {root}: {deleted}")
PY
}

run_roots=()
for repeat in $REPEATS; do
  run_root="$RUNS_BASE/$repeat"
  cases_root="$CASES_ROOT_BASE/$repeat"
  run_roots+=("$run_root")
  clean_failed_outputs "$run_root"

  echo "=== ${MODE}: stability holdout ${repeat} ==="
  echo "cases file: $CASES"
  echo "include cases: $INCLUDE_CASES"
  echo "systems file: $SYSTEMS"
  command=(
    "$BENCHMARK_RUNNER_PYTHON" -m evaluation.layout_benchmark.run_benchmark
    --mode "$MODE" \
    --cases "$CASES" \
    --systems "$SYSTEMS" \
    --include-systems "$INCLUDE_SYSTEMS" \
    --model "$MODEL" \
    --output-root "$run_root" \
    --cases-root "$cases_root" \
    --timeout-s "$TIMEOUT_S"
  )
  if [[ -n "$INCLUDE_CASES" ]]; then
    command+=(--include-cases "$INCLUDE_CASES")
  fi
  "${command[@]}"
done

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  "$BENCHMARK_RUNNER_PYTHON" -m evaluation.layout_benchmark.analyze_stability_holdout \
    --run-roots "${run_roots[@]}" \
    --cases "$CASES" \
    --systems "$SYSTEMS" \
    --output-dir "$ANALYSIS_OUTPUT_DIR"
fi
