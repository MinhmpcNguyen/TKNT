#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run_group() {
  local repeat_root="$1"
  local operation="$2"

  set +e
  TKNT_EDIT_GROUP_OUTPUT_ROOT="${repeat_root}" \
    ./experiments/image_flow_eval/run_edit_operation_group.sh "${operation}" 1
  local status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "Continuing after ${operation} exited ${status}; image eval returns non-zero when metrics fail." >&2
  fi
}

# Recolor repeat 1 is assumed to already exist at:
# experiments/image_flow_eval/runs/edit_manual_by_operation/recolor/batch1
# TKNT_EDIT_GROUP_OUTPUT_ROOT=experiments/image_flow_eval/runs/edit_manual_by_operation_repeat2 \
#   ./experiments/image_flow_eval/run_edit_operation_group.sh recolor 1

for operation in replace remove add; do
  run_group experiments/image_flow_eval/runs/edit_manual_by_operation_repeat1 "${operation}"
  run_group experiments/image_flow_eval/runs/edit_manual_by_operation_repeat2 "${operation}"
done
