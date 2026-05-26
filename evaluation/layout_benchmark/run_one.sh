#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${MODEL:=gpt-5.4-mini}"
: "${TKNT_DISABLE_MODEL_FALLBACK:=1}"
: "${TKNT_ENV_PREFIX:=.conda_envs/benchmark-tknt}"
: "${IDESIGN_ENV_PREFIX:=.conda_envs/benchmark-idesign}"
: "${FLAIRGPT_ENV_PREFIX:=.conda_envs/benchmark-flairgpt}"
: "${RUN_ROOT:=evaluation/layout_benchmark/runs/manual}"
: "${CASES:=evaluation/layout_benchmark/cases.json}"
CASE="${CASE-bedroom_01_rect_door_right_window_left}"
SYSTEM="${SYSTEM-tknt_full}"
: "${CASES_ROOT:=evaluation/layout_benchmark/tknt_case_artifacts/manual}"
: "${TIMEOUT_S:=600}"
: "${EXTERNAL_TIMEOUT_S:=0}"
: "${EXTERNAL_MAX_ATTEMPTS:=3}"
: "${IDESIGN_LLM_TIMEOUT_S:=0}"
: "${FLAIRGPT_LLM_TIMEOUT_S:=0}"

case "$SYSTEM" in
  idesign_gemini)
    SYSTEM="idesign_openai"
    ;;
  flairgpt_gemini)
    SYSTEM="flairgpt_openai"
    ;;
esac

repo_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$value"
  fi
}

TKNT_ENV_PREFIX="$(repo_path "$TKNT_ENV_PREFIX")"
IDESIGN_ENV_PREFIX="$(repo_path "$IDESIGN_ENV_PREFIX")"
FLAIRGPT_ENV_PREFIX="$(repo_path "$FLAIRGPT_ENV_PREFIX")"

: "${TKNT_PYTHON:=$TKNT_ENV_PREFIX/bin/python}"
: "${IDESIGN_PYTHON:=$IDESIGN_ENV_PREFIX/bin/python}"
: "${FLAIRGPT_PYTHON:=$FLAIRGPT_ENV_PREFIX/bin/python}"
: "${BENCHMARK_RUNNER_PYTHON:=$TKNT_PYTHON}"

if [[ ! -x "$BENCHMARK_RUNNER_PYTHON" ]]; then
  echo "Benchmark Python not found: $BENCHMARK_RUNNER_PYTHON" >&2
  echo "Run ./evaluation/layout_benchmark/setup_envs.sh first." >&2
  exit 1
fi

export TKNT_DISABLE_MODEL_FALLBACK
export TKNT_BENCHMARK_TEXT_MODEL="$MODEL"
export TKNT_PYTHON
export IDESIGN_PYTHON
export FLAIRGPT_PYTHON
export IDESIGN_LLM_TIMEOUT_S
export FLAIRGPT_LLM_TIMEOUT_S

cd "$REPO_ROOT"

command=(
  "$BENCHMARK_RUNNER_PYTHON" -m evaluation.layout_benchmark.run_benchmark
  --mode run
  --run-external
  --cases "$CASES"
  --model "$MODEL"
  --output-root "$RUN_ROOT"
  --cases-root "$CASES_ROOT"
  --timeout-s "$TIMEOUT_S"
  --external-timeout-s "$EXTERNAL_TIMEOUT_S"
  --external-max-attempts "$EXTERNAL_MAX_ATTEMPTS"
)

if [[ -n "$CASE" ]]; then
  command+=(--include-cases "$CASE")
fi

if [[ -n "$SYSTEM" ]]; then
  command+=(--include-systems "$SYSTEM")
fi

"${command[@]}"
