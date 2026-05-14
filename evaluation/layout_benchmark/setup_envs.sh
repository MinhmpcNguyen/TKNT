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

repo_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$value"
  fi
}

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  echo "conda was not found. Install Miniconda/Miniforge or put conda on PATH." >&2
  exit 1
fi

TKNT_ENV_PREFIX="$(repo_path "${TKNT_ENV_PREFIX:-.conda_envs/benchmark-tknt}")"
IDESIGN_ENV_PREFIX="$(repo_path "${IDESIGN_ENV_PREFIX:-.conda_envs/benchmark-idesign}")"
FLAIRGPT_ENV_PREFIX="$(repo_path "${FLAIRGPT_ENV_PREFIX:-.conda_envs/benchmark-flairgpt}")"

create_env_if_missing() {
  local prefix="$1"
  local python_version="$2"
  if [[ -x "$prefix/bin/python" ]]; then
    echo "Using existing env: $prefix"
    return
  fi
  echo "Creating env: $prefix"
  "$CONDA_BIN" create -y -p "$prefix" "python=$python_version" pip
}

install_requirements() {
  local prefix="$1"
  local requirements="$2"
  echo "Installing requirements into: $prefix"
  "$CONDA_BIN" run -p "$prefix" python -m pip install --upgrade pip
  "$CONDA_BIN" run -p "$prefix" python -m pip install -r "$requirements"
}

create_env_if_missing "$TKNT_ENV_PREFIX" "3.11"
create_env_if_missing "$IDESIGN_ENV_PREFIX" "3.9"
create_env_if_missing "$FLAIRGPT_ENV_PREFIX" "3.12"

install_requirements "$TKNT_ENV_PREFIX" "$SCRIPT_DIR/requirements/tknt.txt"
install_requirements "$IDESIGN_ENV_PREFIX" "$SCRIPT_DIR/requirements/idesign.txt"
install_requirements "$FLAIRGPT_ENV_PREFIX" "$SCRIPT_DIR/requirements/flairgpt.txt"

cat <<EOF

Benchmark environments are ready.

TKNT_PYTHON=$TKNT_ENV_PREFIX/bin/python
IDESIGN_PYTHON=$IDESIGN_ENV_PREFIX/bin/python
FLAIRGPT_PYTHON=$FLAIRGPT_ENV_PREFIX/bin/python

Run:
  ./evaluation/layout_benchmark/run_one.sh
EOF
