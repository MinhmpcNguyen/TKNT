#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/Yuki/TKNT"
PORT="${1:-8000}"

cd "$ROOT_DIR"

python -m http.server "$PORT"
