#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./experiments/image_flow_eval/run_edit_operation_group.sh <recolor|replace|remove|add> [batch|all]

Examples:
  ./experiments/image_flow_eval/run_edit_operation_group.sh recolor 1
  ./experiments/image_flow_eval/run_edit_operation_group.sh replace 1
  ./experiments/image_flow_eval/run_edit_operation_group.sh remove all
  ./experiments/image_flow_eval/run_edit_operation_group.sh add 2,3

Optional env:
  TKNT_REPLACE_IMAGE=/abs/path/to/object.png
  TKNT_REPLACE_KIND=sofa|wardrobe|bed|desk|chair
  TKNT_ADD_REFERENCE_IMAGE=/abs/path/to/wall_art.jpeg
  TKNT_ADD_PROMPT="Add one small floor lamp inside the selected bbox only..."
  TKNT_CREATE_RUNS_ROOT=experiments/image_flow_eval/runs/first_time
  TKNT_EDIT_GROUP_OUTPUT_ROOT=experiments/image_flow_eval/runs/edit_manual_by_operation
USAGE
}

operation="${1:-}"
batch_arg="${2:-all}"

if [[ -z "${operation}" || "${operation}" == "-h" || "${operation}" == "--help" ]]; then
  usage
  exit 0
fi

case "${operation}" in
  recolor | replace | remove | add) ;;
  *)
    echo "Unknown edit operation: ${operation}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${batch_arg}" == "all" ]]; then
  batches=(1 2 3 4 5)
else
  IFS="," read -r -a batches <<<"${batch_arg}"
fi

if [[ -n "${TKNT_EDIT_MANUAL_MANIFEST_PREFIX:-}" ]]; then
  manual_manifest_prefix="${TKNT_EDIT_MANUAL_MANIFEST_PREFIX}"
elif [[ "${operation}" == "add" ]]; then
  manual_manifest_prefix="${TKNT_EDIT_ADD_MANUAL_MANIFEST_PREFIX:-/private/tmp/tknt_edit_add_manual}"
else
  manual_manifest_prefix="/private/tmp/tknt_edit_manual"
fi
group_manifest_prefix="${TKNT_EDIT_GROUP_MANIFEST_PREFIX:-/private/tmp/tknt_edit_${operation}}"
output_root="${TKNT_EDIT_GROUP_OUTPUT_ROOT:-experiments/image_flow_eval/runs/edit_manual_by_operation}"
create_runs_root="${TKNT_CREATE_RUNS_ROOT:-experiments/image_flow_eval/runs/first_time}"
variants=(${TKNT_EDIT_GROUP_VARIANTS:-edit_prompt_image_only edit_object_scope_preflight})

cd "${REPO_ROOT}"

for batch in "${batches[@]}"; do
  source_manifest="${manual_manifest_prefix}_batch${batch}.json"
  group_manifest="${group_manifest_prefix}_batch${batch}.json"
  batch_output_dir="${output_root}/${operation}/batch${batch}"
  create_batch_dir="${create_runs_root}/curated_core14_create_batch${batch}"

  python - "${operation}" "${source_manifest}" "${group_manifest}" "${batch_output_dir}" "${create_batch_dir}" <<'PY'
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

operation = sys.argv[1]
source_manifest = Path(sys.argv[2])
group_manifest = Path(sys.argv[3])
batch_output_dir = Path(sys.argv[4])
create_batch_dir = Path(sys.argv[5])

edit_variants = [
    {"id": "edit_prompt_image_only", "mode": "edit_prompt_image_only", "provider": "tknt_snapshot"},
    {
        "id": "edit_object_scope_preflight",
        "mode": "edit_object_scope_preflight",
        "provider": "tknt_snapshot",
    },
]


def _prompt_for_operation(name: str, target_name: str) -> str:
    target_phrase = f"selected {target_name}"
    if name == "recolor":
        return (
            f"Change only the {target_phrase} to muted forest green fabric or matte material; "
            "preserve its silhouette, position, scale, perspective, lighting continuity, "
            "handles, seams, drawer lines, panel gaps, decorative motifs, visible "
            "grain/texture structure, foreground occluders, neighboring objects, and "
            "every other object."
        )
    if name == "replace":
        return (
            f"Replace only the {target_phrase} inside bbox_norm with the replacement reference object; "
            "preserve the same bbox footprint, position, scale, perspective, camera, room, "
            "and every other object outside the bbox."
        )
    if name == "remove":
        return (
            f"Remove only the {target_phrase} inside bbox_norm and naturally inpaint the background; "
            "preserve everything outside the bbox exactly, including camera, room shell, furniture, "
            "lighting, and perspective."
        )
    add_prompt = os.environ.get("TKNT_ADD_PROMPT")
    if add_prompt:
        return add_prompt
    return (
        f"Add the {target_name} from the provided reference image inside bbox_norm only; "
        "keep it fully contained in that manually selected placement region. Do not replace, "
        "remove, recolor, move, or cover existing furniture outside the bbox; preserve camera, "
        "room shell, existing furniture, lighting, and everything outside the bbox exactly."
    )


def _replacement_image_for_case(case_id: str) -> str | None:
    override = os.environ.get("TKNT_REPLACE_IMAGE")
    if override:
        return str(Path(override).expanduser().resolve())

    image_dir = Path(os.environ.get("TKNT_REPLACEMENT_IMAGE_DIR", "experiments/a_set_replace_image"))
    image_dir = image_dir.expanduser().resolve()
    files_by_kind = {
        "bed": "bed.png",
        "chair": "chair.jpg",
        "desk": "desk.png",
        "sofa": "sofa.jpeg",
        "wardrobe": "wardrobe.jpg",
    }

    forced_kind = os.environ.get("TKNT_REPLACE_KIND", "").strip().lower()
    if forced_kind:
        filename = files_by_kind.get(forced_kind)
        if filename is None:
            raise ValueError(
                "TKNT_REPLACE_KIND must be one of: "
                + ", ".join(sorted(files_by_kind))
            )
        return str((image_dir / filename).resolve())

    text = case_id.lower()
    if "living" in text:
        filename = "sofa.jpeg"
    elif "common_bedroom_chamfer" in text:
        filename = "chair.jpg"
    elif "desk" in text or "study" in text:
        filename = "desk.png"
    elif "bedroom" in text:
        filename = "wardrobe.jpg"
    else:
        filename = "sofa.jpeg"
    return str((image_dir / filename).resolve())


def _add_reference_image() -> str:
    raw_path = os.environ.get(
        "TKNT_ADD_REFERENCE_IMAGE",
        "experiments/a_set_replace_image/wall_art.jpeg",
    )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "Add reference image not found. Expected wall art reference at: "
            f"{resolved}"
        )
    return str(resolved)


def _add_object_id() -> str:
    return os.environ.get("TKNT_ADD_OBJECT_ID", "manual_add_box_1").strip() or "manual_add_box_1"


def _add_object_name() -> str:
    return os.environ.get("TKNT_ADD_OBJECT_NAME", "wall art").strip() or "wall art"


def _target_name_for_case(case: dict) -> str:
    override = os.environ.get("TKNT_EDIT_TARGET_NAME")
    if override:
        return _label(override)

    operations = case.get("edit_operations") or []
    first_operation = operations[0] if operations and isinstance(operations[0], dict) else {}
    bbox = first_operation.get("bbox_norm")
    snapshot_payload = case.get("snapshot_payload")
    if isinstance(bbox, dict) and isinstance(snapshot_payload, str):
        matched = _target_name_from_snapshot_payload(Path(snapshot_payload), bbox)
        if matched:
            return matched

    source_variant = case.get("edit_source_variant") or {}
    source_case_id = source_variant.get("case_id")
    if not isinstance(source_case_id, str) or not source_case_id:
        source_case_id = str(case.get("id", "")).removesuffix("__edit_manual")
    return _target_name_from_case_id(source_case_id)


def _target_name_from_snapshot_payload(payload_path: Path, bbox: dict) -> str | None:
    try:
        payload = json.loads(payload_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    image_width = _number((payload.get("canvas") or {}).get("widthPx"))
    image_height = _number((payload.get("canvas") or {}).get("heightPx"))
    if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
        return None

    target_bbox = _normalized_bbox(bbox)
    if target_bbox is None:
        return None

    best_label = None
    best_score = 0.0
    for item in payload.get("visibleObjects") or []:
        if not isinstance(item, dict):
            continue
        screen_bbox = item.get("screenBboxPx")
        if not isinstance(screen_bbox, dict):
            continue
        candidate_bbox = _normalized_screen_bbox(screen_bbox, image_width, image_height)
        if candidate_bbox is None:
            continue
        iou = _bbox_iou(target_bbox, candidate_bbox)
        center_score = max(0.0, 1.0 - _center_distance(target_bbox, candidate_bbox) * 4.0)
        area_ratio = _area_similarity(target_bbox, candidate_bbox)
        score = iou * 2.0 + center_score * 0.35 + area_ratio * 0.15
        if score > best_score:
            raw_label = item.get("type") or item.get("id") or item.get("assetId")
            best_label = _label(str(raw_label)) if raw_label else None
            best_score = score
    return best_label if best_score >= 0.2 else None


def _target_name_from_case_id(case_id: str) -> str:
    text = case_id.lower()
    if "living" in text:
        return "sofa"
    if "common_bedroom_chamfer" in text:
        return "chair"
    if "desk" in text or "study" in text:
        return "desk"
    if "bedroom" in text:
        return "wardrobe"
    return "object"


def _label(value: str) -> str:
    label = value.strip().replace("_", " ").replace("-", " ")
    return " ".join(label.split()) or "object"


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalized_bbox(value: dict) -> tuple[float, float, float, float] | None:
    x = _number(value.get("x"))
    y = _number(value.get("y"))
    width = _number(value.get("width"))
    height = _number(value.get("height"))
    if x is None or y is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    return _clamp_bbox(x, y, x + width, y + height)


def _normalized_screen_bbox(
    value: dict,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    min_x = _number(value.get("minX"))
    min_y = _number(value.get("minY"))
    max_x = _number(value.get("maxX"))
    max_y = _number(value.get("maxY"))
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    return _clamp_bbox(
        min_x / image_width,
        min_y / image_height,
        max_x / image_width,
        max_y / image_height,
    )


def _clamp_bbox(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> tuple[float, float, float, float] | None:
    left = min(max(min_x, 0.0), 1.0)
    top = min(max(min_y, 0.0), 1.0)
    right = min(max(max_x, 0.0), 1.0)
    bottom = min(max(max_y, 0.0), 1.0)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    inter_left = max(left[0], right[0])
    inter_top = max(left[1], right[1])
    inter_right = min(left[2], right[2])
    inter_bottom = min(left[3], right[3])
    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return inter_area / max(left_area + right_area - inter_area, 1e-6)


def _center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x = (left[0] + left[2]) / 2.0
    left_y = (left[1] + left[3]) / 2.0
    right_x = (right[0] + right[2]) / 2.0
    right_y = (right[1] + right[3]) / 2.0
    return ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5


def _area_similarity(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    if left_area <= 0 or right_area <= 0:
        return 0.0
    return min(left_area, right_area) / max(left_area, right_area)


def _ensure_create_case_link(case_id: str) -> None:
    create_case_dir = (create_batch_dir / case_id).resolve()
    if not create_case_dir.is_dir():
        raise FileNotFoundError(f"Create case dir not found: {create_case_dir}")

    batch_output_dir.mkdir(parents=True, exist_ok=True)
    link_path = batch_output_dir / case_id
    if link_path.is_symlink():
        if link_path.resolve() == create_case_dir:
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.resolve() == create_case_dir:
            return
        raise FileExistsError(f"Cannot create source link; path already exists: {link_path}")
    link_path.symlink_to(create_case_dir, target_is_directory=True)


def _create_bbox_guide(
    *,
    edit_case_id: str,
    source_case_id: str,
    source_variant_name: str,
    operation_payload: dict,
) -> str:
    render_path = create_batch_dir / source_case_id / source_variant_name / "render.png"
    if not render_path.is_file():
        raise FileNotFoundError(f"Approved render for bbox guide not found: {render_path}")

    bbox = operation_payload.get("bbox_norm")
    if not isinstance(bbox, dict):
        raise ValueError(f"Operation has no bbox_norm for bbox guide: {edit_case_id}")

    image = Image.open(render_path).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (15, 23, 42, 28))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image, "RGBA")

    left = int(float(bbox["x"]) * width)
    top = int(float(bbox["y"]) * height)
    right = int((float(bbox["x"]) + float(bbox["width"])) * width)
    bottom = int((float(bbox["y"]) + float(bbox["height"])) * height)
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox for bbox guide: {edit_case_id}")

    color = (249, 115, 22, 255)
    line_width = max(4, round(min(width, height) * 0.006))
    draw.rectangle(
        (left, top, right, bottom),
        fill=(249, 115, 22, 36),
        outline=color,
        width=line_width,
    )
    label = f"{operation_payload.get('object_id', 'manual_box')} {operation}"
    font = ImageFont.load_default()
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 18
    label_height = label_box[3] - label_box[1] + 14
    label_left = max(6, min(left + 6, max(6, width - label_width - 6)))
    label_top = max(6, min(top + 6, max(6, height - label_height - 6)))
    draw.rectangle(
        (
            label_left,
            label_top,
            label_left + label_width,
            label_top + label_height,
        ),
        fill=(255, 255, 255, 245),
        outline=color,
        width=2,
    )
    draw.text(
        (label_left + 9, label_top + 7),
        label,
        fill=(17, 24, 39, 255),
        font=font,
    )

    safe_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in edit_case_id)
    guide_path = batch_output_dir / "_bbox_guides" / f"{safe_id}__{operation}_bbox.png"
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(guide_path)
    return str(guide_path.resolve())


def _rewrite_case(case: dict) -> dict:
    rewritten = deepcopy(case)
    source_variant = rewritten.get("edit_source_variant") or {}
    source_case_id = source_variant.get("case_id")
    if not isinstance(source_case_id, str) or not source_case_id:
        source_case_id = str(rewritten["id"]).removesuffix("__edit_manual")
    source_variant_name = source_variant.get("variant")
    if not isinstance(source_variant_name, str) or not source_variant_name:
        source_variant_name = "single_edit_edge_silhouette_guided"
    _ensure_create_case_link(source_case_id)

    target_name = _add_object_name() if operation == "add" else _target_name_for_case(rewritten)
    prompt = _prompt_for_operation(operation, target_name)
    if operation == "replace":
        replacement_image = _replacement_image_for_case(source_case_id)
    elif operation == "add":
        replacement_image = _add_reference_image()
    else:
        replacement_image = None
    target_color = "muted forest green" if operation == "recolor" else None

    operations = rewritten.get("edit_operations") or []
    if not isinstance(operations, list) or not operations:
        raise ValueError(f"Case has no edit_operations: {rewritten.get('id')}")

    new_operations = []
    for item in operations:
        new_item = dict(item)
        if "bbox_norm" not in new_item or new_item["bbox_norm"] is None:
            raise ValueError(f"Operation has no bbox_norm: {rewritten.get('id')}")
        new_item["operation_type"] = operation
        if operation == "add":
            new_item["object_id"] = _add_object_id()
        new_item["object_name"] = target_name
        new_item["prompt"] = prompt
        new_item["replacement_image"] = replacement_image
        new_item["target_color"] = target_color
        new_operations.append(new_item)

    rewritten["edit_operations"] = new_operations
    rewritten["edit_prompt"] = prompt
    rewritten["annotated_reference_image"] = _create_bbox_guide(
        edit_case_id=str(rewritten.get("id", source_case_id)),
        source_case_id=source_case_id,
        source_variant_name=source_variant_name,
        operation_payload=new_operations[0],
    )
    rewritten["variants"] = list(edit_variants)
    return rewritten


if not source_manifest.is_file():
    if operation == "add":
        raise FileNotFoundError(
            "Add bbox manifest not found: "
            f"{source_manifest}. Draw add placement bboxes first with "
            "./experiments/image_flow_eval/annotate_add_bboxes.sh and click Write manifests."
        )
    raise FileNotFoundError(f"Manual bbox manifest not found: {source_manifest}")
payload = json.loads(source_manifest.read_text())
cases = payload.get("cases")
if not isinstance(cases, list):
    raise ValueError(f"Manifest has no cases list: {source_manifest}")

payload["cases"] = [_rewrite_case(case) for case in cases]
payload["variants"] = list(edit_variants)

group_manifest.parent.mkdir(parents=True, exist_ok=True)
group_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
print(
    f"Prepared {operation} batch manifest: {group_manifest} "
    f"({len(payload['cases'])} case(s)); output={batch_output_dir}"
)
PY

  if [[ "${TKNT_EDIT_GROUP_PREPARE_ONLY:-0}" == "1" ]]; then
    continue
  fi

  ./experiments/image_flow_eval/run_image_flow_eval.sh run \
    --manifest "${group_manifest}" \
    --output-dir "${batch_output_dir}" \
    --provider tknt_snapshot \
    --resume \
    --variants "${variants[@]}"
done
