#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${REPO_ROOT}/.cache/ultralytics}"
export TKNT_IMAGE_FLOW_METRIC_DEVICE="${TKNT_IMAGE_FLOW_METRIC_DEVICE:-cpu}"
export TKNT_IMAGE_FLOW_DINO_MODEL="${TKNT_IMAGE_FLOW_DINO_MODEL:-facebook/dinov2-small}"
export TKNT_IMAGE_FLOW_YOLO_MODEL="${TKNT_IMAGE_FLOW_YOLO_MODEL:-${REPO_ROOT}/models/metrics/yolo11n.pt}"

mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${MPLCONFIGDIR}" "${YOLO_CONFIG_DIR}"

PYTHON_BIN="${TKNT_IMAGE_FLOW_PYTHON:-${REPO_ROOT}/.conda_envs/prj-tknt/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m experiments.image_flow_eval.image_flow_eval.cli "$@"
