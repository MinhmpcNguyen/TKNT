from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from experiments.image_flow_eval.image_flow_eval.io import write_json_file
from experiments.image_flow_eval.image_flow_eval.types import JSONValue

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_AUDIT_ROOT = _REPO_ROOT / "experiments" / "image_flow_eval" / "snapshot_audit"
_RUNS_ROOT = _REPO_ROOT / "experiments" / "image_flow_eval" / "runs"
_SELECTED_CASES_PATH = _SNAPSHOT_AUDIT_ROOT / "selected_snapshot_cases.json"
_SELECTED_CASE_SECTIONS = (
    "primary_existing_cases",
    "primary_hard_cases",
    "secondary_hard_cases",
)
_CORE_CASE_IDS = (
    "tknt_new_s_o_simple_bedroom_02_master_reading_tknt_full",
    "tknt_new_s_o_simple_bedroom_04_guest_vanity_tknt_full",
    "tknt_new_s_o_simple_living_02_compact_reading_tknt_full",
    "tknt_new_s_o_simple_living_04_hosting_lounge_tknt_full",
    "tknt_new_s_wo_simple_living_03_family_media_tknt_full",
    "tknt_new_hard_nonrect_common_living_chamfer_openings_tknt_full",
    "tknt_new_hard_nonrect_proof_living_l_openings_tknt_full",
    "tknt_new_hard_nonrect_common_bedroom_chamfer_openings_tknt_full",
    "tknt_new_hard_nonrect_proof_bedroom_l_openings_tknt_full",
    "tknt_new_s_o_simple_bedroom_01_rest_study_tknt_full",
    "tknt_new_hard_nonrect_common_living_bay_window_openings_tknt_full",
    "tknt_new_hard_living_03_l_shape_circulation_tknt_full",
    "tknt_new_hard_bedroom_05_l_shape_storage_tknt_full",
    "tknt_new_hard_nonrect_common_bedroom_side_recess_openings_tknt_full",
)
_PRESET_FOLLOW_CASE_IDS = (
    "tknt_new_s_o_simple_bedroom_04_guest_vanity_tknt_full",
    "tknt_new_s_o_simple_living_02_compact_reading_tknt_full",
    "tknt_new_hard_nonrect_common_living_chamfer_openings_tknt_full",
    "tknt_new_hard_nonrect_proof_bedroom_l_openings_tknt_full",
)
_NO_SCENERY_CASE_IDS = frozenset(
    {
        "tknt_new_s_o_simple_bedroom_02_master_reading_tknt_full",
        "tknt_new_s_wo_simple_living_03_family_media_tknt_full",
        "tknt_new_hard_nonrect_common_bedroom_chamfer_openings_tknt_full",
        "tknt_new_hard_nonrect_proof_bedroom_l_openings_tknt_full",
    }
)
_CREATE_VARIANTS: list[JSONValue] = [
    {
        "id": "contract_objects_preflight_locked_geometry",
        "mode": "contract_objects_preflight_locked_geometry",
        "provider": "tknt_snapshot",
    },
    {
        "id": "llm_visual_preflight",
        "mode": "llm_visual_preflight",
        "provider": "tknt_snapshot",
    },
    {
        "id": "llm_visual_preflight_object_anchors",
        "mode": "llm_visual_preflight_object_anchors",
        "provider": "tknt_snapshot",
    },
    {
        "id": "single_edit_edge_silhouette_guided",
        "mode": "single_edit_edge_silhouette_guided",
        "provider": "tknt_snapshot",
    },
    {
        "id": "contract_objects_preflight",
        "mode": "contract_objects_preflight",
        "provider": "tknt_snapshot",
    },
    {
        "id": "contract_objects",
        "mode": "contract_objects",
        "provider": "tknt_snapshot",
    },
    {
        "id": "simple_prompt",
        "mode": "simple_prompt",
        "provider": "tknt_snapshot",
    },
]
_CREATE_METRICS: list[JSONValue] = [
    "blank",
    "layout_fidelity",
    "layout_gate_score",
    "layout_edge_structure",
    "layout_depth_structure",
    "layout_dino_patch_similarity",
    "layout_dino_object_similarity",
    "structural_layout_score",
    "detector_layout",
    "detector_spatial_relations",
    "preset_fidelity",
]
_RENDER_PROMPT = (
    "Photorealize the exact 3D layout. Preserve camera, room shell, openings, "
    "furniture count, object positions, object scale, and object colors; only "
    "improve realism, material finish, selected style, and selected lighting. "
    "If a scenery preset is selected, apply it only through existing windows."
)
_BATCH_SIZE = 3


@dataclass(frozen=True)
class SelectedSnapshotCase:
    case_id: str
    snapshot_image: Path
    snapshot_payload: Path


@dataclass(frozen=True)
class PresetPack:
    id: str
    style: str
    lighting: str
    scenery: str | None

    def to_mapping(self) -> dict[str, JSONValue]:
        mapping: dict[str, JSONValue] = {
            "style": self.style,
            "lighting": self.lighting,
        }
        if self.scenery:
            mapping["scenery"] = self.scenery
        return mapping

    def without_scenery(self) -> PresetPack:
        return PresetPack(
            id=f"{self.id}_no_scenery",
            style=self.style,
            lighting=self.lighting,
            scenery=None,
        )


_PRESET_PACKS = (
    PresetPack(
        id="japandi_day_garden",
        style="japandi",
        lighting="daylight_6",
        scenery="garden",
    ),
    PresetPack(
        id="modern_night_city",
        style="modern",
        lighting="night_3",
        scenery="city",
    ),
    PresetPack(
        id="industrial_overcast_mountain",
        style="industrial",
        lighting="daylight_4",
        scenery="mountain",
    ),
)
_DEFAULT_PRESET = _PRESET_PACKS[0]


def build_core_manifest(
    selected_cases_path: Path = _SELECTED_CASES_PATH,
    *,
    output_dir: Path = _RUNS_ROOT / "curated_core14_create",
) -> dict[str, JSONValue]:
    selected_cases = _load_selected_cases(selected_cases_path)
    return _build_manifest(
        cases=[
            _build_case_entry(
                _required_selected_case(selected_cases, case_id),
                preset_selection=_core_case_preset_override(case_id),
            )
            for case_id in _CORE_CASE_IDS
        ],
        output_dir=output_dir,
        default_preset=_DEFAULT_PRESET,
    )


def build_core_batch_manifests(
    selected_cases_path: Path = _SELECTED_CASES_PATH,
    *,
    runs_root: Path = _RUNS_ROOT,
    batch_size: int = _BATCH_SIZE,
) -> list[tuple[str, dict[str, JSONValue]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    selected_cases = _load_selected_cases(selected_cases_path)
    core_cases = [
        _build_case_entry(
            _required_selected_case(selected_cases, case_id),
            preset_selection=_core_case_preset_override(case_id),
        )
        for case_id in _CORE_CASE_IDS
    ]
    manifests: list[tuple[str, dict[str, JSONValue]]] = []
    for offset in range(0, len(core_cases), batch_size):
        batch_number = offset // batch_size + 1
        name = f"manifest.curated_core14_create.batch{batch_number}.json"
        manifests.append(
            (
                name,
                _build_manifest(
                    cases=core_cases[offset : offset + batch_size],
                    output_dir=runs_root / f"curated_core14_create_batch{batch_number}",
                    default_preset=_DEFAULT_PRESET,
                ),
            )
        )
    return manifests


def build_preset_follow_manifest(
    selected_cases_path: Path = _SELECTED_CASES_PATH,
    *,
    output_dir: Path = _RUNS_ROOT / "curated_preset_follow_create",
    preset_packs: Sequence[PresetPack] = _PRESET_PACKS,
) -> dict[str, JSONValue]:
    selected_cases = _load_selected_cases(selected_cases_path)
    return _build_manifest(
        cases=[
            _build_case_entry(
                _required_selected_case(selected_cases, case_id),
                case_id=f"{case_id}__preset_{preset_pack.id}",
                preset_selection=_case_preset_selection(
                    case_id=case_id,
                    preset_pack=preset_pack,
                ),
            )
            for preset_pack in preset_packs
            for case_id in _PRESET_FOLLOW_CASE_IDS
        ],
        output_dir=output_dir,
        default_preset=None,
    )


def build_preset_follow_pack_manifests(
    selected_cases_path: Path = _SELECTED_CASES_PATH,
    *,
    runs_root: Path = _RUNS_ROOT,
    preset_packs: Sequence[PresetPack] = _PRESET_PACKS,
) -> list[tuple[str, dict[str, JSONValue]]]:
    selected_cases = _load_selected_cases(selected_cases_path)
    return [
        (
            f"manifest.curated_preset_follow_create.{preset_pack.id}.json",
            _build_manifest(
                cases=[
                    _build_case_entry(
                        _required_selected_case(selected_cases, case_id),
                        preset_selection=_case_preset_selection(
                            case_id=case_id,
                            preset_pack=preset_pack,
                        ),
                    )
                    for case_id in _PRESET_FOLLOW_CASE_IDS
                ],
                output_dir=runs_root / f"curated_preset_follow_create_{preset_pack.id}",
                default_preset=preset_pack,
            ),
        )
        for preset_pack in preset_packs
    ]


def write_curated_manifests(
    selected_cases_path: Path = _SELECTED_CASES_PATH,
    *,
    output_root: Path = _SNAPSHOT_AUDIT_ROOT,
    runs_root: Path = _RUNS_ROOT,
) -> list[Path]:
    manifests = [
        (
            "manifest.curated_core14_create.json",
            build_core_manifest(
                selected_cases_path,
                output_dir=runs_root / "curated_core14_create",
            ),
        ),
        *build_core_batch_manifests(
            selected_cases_path,
            runs_root=runs_root,
        ),
        (
            "manifest.curated_preset_follow_create.json",
            build_preset_follow_manifest(
                selected_cases_path,
                output_dir=runs_root / "curated_preset_follow_create",
            ),
        ),
        *build_preset_follow_pack_manifests(
            selected_cases_path,
            runs_root=runs_root,
        ),
    ]
    written_paths: list[Path] = []
    for name, manifest in manifests:
        output_path = output_root / name
        write_json_file(output_path, manifest)
        written_paths.append(output_path)
        logger.info("Wrote curated image-flow manifest: %s", output_path)
    return written_paths


def _build_manifest(
    *,
    cases: list[JSONValue],
    output_dir: Path,
    default_preset: PresetPack | None,
) -> dict[str, JSONValue]:
    defaults: dict[str, JSONValue] = {
        "metrics": _CREATE_METRICS,
        "align_mode": "resize_to_reference",
        "thresholds": {
            "min_preset_fidelity_score": 0.75,
            "max_layout_aspect_delta": 0.015,
            "min_layout_fidelity_score": 0.82,
            "min_layout_gate_score": 0.65,
            "min_layout_edge_structure_score": 0.35,
            "min_layout_depth_structure_score": 0.45,
            "min_layout_dino_patch_similarity": 0.50,
            "min_structural_layout_score": 0.58,
        },
    }
    if default_preset is not None:
        defaults["preset_selection"] = default_preset.to_mapping()
    return {
        "defaults": defaults,
        "output_dir": str(output_dir),
        "cases": cases,
    }


def _build_case_entry(
    selected_case: SelectedSnapshotCase,
    *,
    case_id: str | None = None,
    preset_selection: PresetPack | None = None,
) -> dict[str, JSONValue]:
    entry: dict[str, JSONValue] = {
        "id": case_id or selected_case.case_id,
        "snapshot_payload": str(selected_case.snapshot_payload),
        "snapshot_image": str(selected_case.snapshot_image),
        "render_prompt": _RENDER_PROMPT,
        "variants": _CREATE_VARIANTS,
    }
    if preset_selection is not None:
        entry["preset_selection"] = preset_selection.to_mapping()
    return entry


def _core_case_preset_override(case_id: str) -> PresetPack | None:
    if case_id not in _NO_SCENERY_CASE_IDS:
        return None
    return _DEFAULT_PRESET.without_scenery()


def _case_preset_selection(
    *,
    case_id: str,
    preset_pack: PresetPack,
) -> PresetPack:
    if case_id in _NO_SCENERY_CASE_IDS:
        return preset_pack.without_scenery()
    return preset_pack


def _load_selected_cases(path: Path) -> Mapping[str, SelectedSnapshotCase]:
    payload = _read_mapping(path)
    selected_cases: dict[str, SelectedSnapshotCase] = {}
    for section_name in _SELECTED_CASE_SECTIONS:
        for raw_item in _sequence(payload.get(section_name)):
            item = _mapping(raw_item)
            case_id = _required_string(item, "case_id", context=section_name)
            selected_case = SelectedSnapshotCase(
                case_id=case_id,
                snapshot_image=_required_file(item, "snapshot_image", context=case_id),
                snapshot_payload=_required_file(
                    item,
                    "snapshot_payload",
                    context=case_id,
                ),
            )
            if case_id in selected_cases:
                raise ValueError(f"Duplicate selected snapshot case: {case_id}")
            selected_cases[case_id] = selected_case
    return selected_cases


def _required_selected_case(
    selected_cases: Mapping[str, SelectedSnapshotCase],
    case_id: str,
) -> SelectedSnapshotCase:
    selected_case = selected_cases.get(case_id)
    if selected_case is None:
        raise ValueError(f"Selected snapshot case is missing: {case_id}")
    return selected_case


def _required_file(
    payload: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> Path:
    path = (
        Path(_required_string(payload, field, context=context)).expanduser().resolve()
    )
    if not path.is_file():
        raise ValueError(f"Selected snapshot file is missing for {context}: {path}")
    return path


def _required_string(
    payload: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string {context}.{field}.")
    return value.strip()


def _read_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    raise ValueError("Expected selected snapshot case object.")


def _sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    raise ValueError("Expected selected snapshot case list.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build curated core and preset-follow create manifests."
    )
    parser.add_argument(
        "--selected-cases",
        type=Path,
        default=_SELECTED_CASES_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_SNAPSHOT_AUDIT_ROOT,
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_RUNS_ROOT,
    )
    args = parser.parse_args()
    written_paths = write_curated_manifests(
        cast(Path, args.selected_cases).expanduser().resolve(),
        output_root=cast(Path, args.output_root).expanduser().resolve(),
        runs_root=cast(Path, args.runs_root).expanduser().resolve(),
    )
    logger.info("Wrote %s curated manifest(s).", len(written_paths))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
