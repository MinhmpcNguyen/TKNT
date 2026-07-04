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
REQUESTED_REPEAT_ID="${REPEAT_ID:-}"
REQUESTED_RUNS_BASE="${RUNS_BASE:-}"
REQUESTED_CASES_ROOT_BASE="${CASES_ROOT_BASE:-}"
REQUESTED_SYSTEMS="${SYSTEMS:-}"
REQUESTED_INCLUDE_SYSTEMS="${INCLUDE_SYSTEMS:-}"
REQUESTED_TIMEOUT_S="${TIMEOUT_S:-}"
REQUESTED_CLEAN_FAILED_OUTPUTS="${CLEAN_FAILED_OUTPUTS:-}"
REQUESTED_POSTGRES_HOST="${POSTGRES_HOST:-}"
REQUESTED_POSTGRES_PORT="${POSTGRES_PORT:-}"
REQUESTED_POSTGRES_DB="${POSTGRES_DB:-}"
REQUESTED_POSTGRES_USER="${POSTGRES_USER:-}"
REQUESTED_POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
REQUESTED_HARD_CASES="${HARD_CASES:-}"
REQUESTED_SIMPLE_CASES="${SIMPLE_CASES:-}"
REQUESTED_SIMPLE_OPENINGS_CASES="${SIMPLE_OPENINGS_CASES:-}"
REQUESTED_SIMPLE_NO_OPENINGS_CASES="${SIMPLE_NO_OPENINGS_CASES:-}"
REQUESTED_SIMPLE_SIMPLE_OPENINGS_CASES="${SIMPLE_SIMPLE_OPENINGS_CASES:-}"
REQUESTED_SIMPLE_SIMPLE_NO_OPENINGS_CASES="${SIMPLE_SIMPLE_NO_OPENINGS_CASES:-}"
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
[[ -n "$REQUESTED_REPEAT_ID" ]] && REPEAT_ID="$REQUESTED_REPEAT_ID"
[[ -n "$REQUESTED_RUNS_BASE" ]] && RUNS_BASE="$REQUESTED_RUNS_BASE"
[[ -n "$REQUESTED_CASES_ROOT_BASE" ]] && CASES_ROOT_BASE="$REQUESTED_CASES_ROOT_BASE"
[[ -n "$REQUESTED_SYSTEMS" ]] && SYSTEMS="$REQUESTED_SYSTEMS"
[[ -n "$REQUESTED_INCLUDE_SYSTEMS" ]] && INCLUDE_SYSTEMS="$REQUESTED_INCLUDE_SYSTEMS"
[[ -n "$REQUESTED_TIMEOUT_S" ]] && TIMEOUT_S="$REQUESTED_TIMEOUT_S"
[[ -n "$REQUESTED_CLEAN_FAILED_OUTPUTS" ]] && CLEAN_FAILED_OUTPUTS="$REQUESTED_CLEAN_FAILED_OUTPUTS"
[[ -n "$REQUESTED_HARD_CASES" ]] && HARD_CASES="$REQUESTED_HARD_CASES"
[[ -n "$REQUESTED_SIMPLE_CASES" ]] && SIMPLE_CASES="$REQUESTED_SIMPLE_CASES"
[[ -n "$REQUESTED_SIMPLE_OPENINGS_CASES" ]] && SIMPLE_OPENINGS_CASES="$REQUESTED_SIMPLE_OPENINGS_CASES"
[[ -n "$REQUESTED_SIMPLE_NO_OPENINGS_CASES" ]] && SIMPLE_NO_OPENINGS_CASES="$REQUESTED_SIMPLE_NO_OPENINGS_CASES"
[[ -n "$REQUESTED_SIMPLE_SIMPLE_OPENINGS_CASES" ]] && SIMPLE_SIMPLE_OPENINGS_CASES="$REQUESTED_SIMPLE_SIMPLE_OPENINGS_CASES"
[[ -n "$REQUESTED_SIMPLE_SIMPLE_NO_OPENINGS_CASES" ]] && SIMPLE_SIMPLE_NO_OPENINGS_CASES="$REQUESTED_SIMPLE_SIMPLE_NO_OPENINGS_CASES"
[[ -n "$REQUESTED_BENCHMARK_RUNNER_PYTHON" ]] && BENCHMARK_RUNNER_PYTHON="$REQUESTED_BENCHMARK_RUNNER_PYTHON"

: "${MODEL:=gpt-5.4-mini}"
: "${MODE:=resume}"
: "${TKNT_DISABLE_MODEL_FALLBACK:=1}"
TKNT_ENV_PREFIX="${REQUESTED_TKNT_ENV_PREFIX:-.conda_envs/prj-tknt}"
: "${REPEAT_ID:=}"
: "${RUNS_BASE:=evaluation/layout_benchmark/runs/manual_ablation/runs}"
: "${CASES_ROOT_BASE:=evaluation/layout_benchmark/tknt_case_artifacts/manual_ablation}"
: "${SYSTEMS:=evaluation/layout_benchmark/systems_ablation.json}"
: "${INCLUDE_SYSTEMS:=tknt_ab_minimal_room_context,tknt_ab_no_style_policy,tknt_ab_no_semantic_layout_planner,tknt_ab_no_request_contract,tknt_ab_no_tier_capacity,tknt_ab_single_concept,tknt_ab_no_concept_repair,tknt_ab_no_material_macro_repair,tknt_ab_no_accessory_refill,tknt_ab_no_stylist_llm,tknt_ab_no_gallery_selector}"
: "${TIMEOUT_S:=1200}"
: "${CLEAN_FAILED_OUTPUTS:=0}"
: "${HARD_CASES:=bedroom_05_l_shape_storage,living_03_l_shape_circulation,nonrect_common_bedroom_chamfer_openings,nonrect_common_living_bay_window_openings}"
: "${SIMPLE_CASES:=simple_living_02_compact_reading,simple_bedroom_01_rest_study}"
: "${SIMPLE_OPENINGS_CASES:=simple_living_04_hosting_lounge,simple_bedroom_03_student_storage}"
: "${SIMPLE_NO_OPENINGS_CASES:=simple_living_05_cozy_apartment,simple_bedroom_02_master_reading}"
: "${SIMPLE_SIMPLE_OPENINGS_CASES:=simple_simple_living_02_3m4x4m2,simple_simple_bedroom_04_3m6x3m8}"
: "${SIMPLE_SIMPLE_NO_OPENINGS_CASES:=simple_simple_living_01_4m0x5m0,simple_simple_bedroom_03_3m2x3m6}"

if [[ -n "$REPEAT_ID" && -z "$REQUESTED_RUNS_BASE" ]]; then
  RUNS_BASE="evaluation/layout_benchmark/runs/manual_ablation_${REPEAT_ID}/runs"
fi
if [[ -n "$REPEAT_ID" && -z "$REQUESTED_CASES_ROOT_BASE" ]]; then
  CASES_ROOT_BASE="evaluation/layout_benchmark/tknt_case_artifacts/manual_ablation_${REPEAT_ID}"
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
    except Exception:
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

run_case_set() {
  local name="$1"
  local cases_file="$2"
  local include_cases="$3"
  local run_root="$RUNS_BASE/$name"
  local cases_root="$CASES_ROOT_BASE/$name"

  clean_failed_outputs "$run_root"
  echo "=== ${MODE}: ${name} (${cases_file}) ==="
  echo "cases: ${include_cases}"

  local command=(
    "$BENCHMARK_RUNNER_PYTHON" -m evaluation.layout_benchmark.run_benchmark
    --mode "$MODE" \
    --cases "evaluation/layout_benchmark/$cases_file" \
    --systems "$SYSTEMS" \
    --include-systems "$INCLUDE_SYSTEMS" \
    --model "$MODEL" \
    --output-root "$run_root" \
    --cases-root "$cases_root" \
    --timeout-s "$TIMEOUT_S"
  )
  if [[ -n "$include_cases" ]]; then
    command+=(--include-cases "$include_cases")
  fi

  "${command[@]}"
}

run_case_set "tknt_new_hard" "cases.json" "$HARD_CASES"
run_case_set "tknt_new_s" "simple_cases.json" "$SIMPLE_CASES"
run_case_set "tknt_new_s_o" "simple_cases_with_openings.json" "$SIMPLE_OPENINGS_CASES"
run_case_set "tknt_new_s_wo" "simple_cases_without_openings.json" "$SIMPLE_NO_OPENINGS_CASES"
run_case_set "tknt_new_ss_o" "simple_simple_cases_with_openings.json" "$SIMPLE_SIMPLE_OPENINGS_CASES"
run_case_set "tknt_new_ss_wo" "simple_simple_cases_without_openings.json" "$SIMPLE_SIMPLE_NO_OPENINGS_CASES"
