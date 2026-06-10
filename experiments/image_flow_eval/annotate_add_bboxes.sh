#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

python experiments/image_flow_eval/annotate_edit_bboxes.py \
  --annotation-path experiments/image_flow_eval/snapshot_audit/edit_add_bbox_annotations.json \
  --manifest-output-dir /private/tmp/tknt_edit_add_manual \
  --output-root experiments/image_flow_eval/runs/edit_manual_add_bbox \
  --default-operation-type add \
  --default-object-id manual_add_box_1 \
  --default-object-name "wall art" \
  --default-prompt "Add the wall art from the provided reference image inside the selected bbox only; keep it contained in that placement region and preserve camera, room shell, existing furniture, lighting, and everything outside the bbox exactly." \
  "$@"
