from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.image_flow_eval.annotate_edit_bboxes import (
    ManualEditAnnotation,
    bbox_norm_from_points,
    build_manual_edit_manifests,
    create_edit_guide_image,
    load_edit_targets,
    load_manual_edit_batches,
    manifest_output_path_for_batch,
)
from experiments.image_flow_eval.image_flow_eval.io import load_manifest


class ManualEditBBoxToolTest(unittest.TestCase):
    def test_bbox_norm_from_points_clamps_and_orders_coordinates(self) -> None:
        bbox = bbox_norm_from_points(0.8, -0.2, 0.25, 0.6)

        self.assertEqual(
            bbox,
            {"x": 0.25, "y": 0.0, "width": 0.55, "height": 0.6},
        )

    def test_manifest_output_path_uses_planned_prefix_shape(self) -> None:
        self.assertEqual(
            manifest_output_path_for_batch(Path("/private/tmp/tknt_edit_manual"), 3),
            Path("/private/tmp/tknt_edit_manual_batch3.json"),
        )

    def test_guide_image_is_written_at_render_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_path = root / "render.png"
            Image.new("RGB", (80, 40), "white").save(render_path)
            guide_path = root / "guide.png"

            create_edit_guide_image(
                source_image_path=render_path,
                output_path=guide_path,
                annotation=_annotation(),
            )

            self.assertTrue(guide_path.is_file())
            with Image.open(guide_path) as image:
                self.assertEqual(image.size, (80, 40))

    def test_load_edit_targets_sorts_recommended_objects_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "snapshot_payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 200, "heightPx": 100},
                        "visibleObjects": [
                            {
                                "id": "rug_001",
                                "type": "rug",
                                "screenBboxPx": {
                                    "minX": 0,
                                    "minY": 70,
                                    "width": 180,
                                    "height": 20,
                                },
                            },
                            {
                                "id": "sofa",
                                "type": "sofa",
                                "screenBboxPx": {
                                    "minX": 60,
                                    "minY": 30,
                                    "width": 50,
                                    "height": 30,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            targets = load_edit_targets(payload_path)

        self.assertEqual(targets[0].object_id, "sofa")
        self.assertTrue(targets[0].recommended)
        self.assertEqual(
            targets[0].bbox_norm,
            {"x": 0.3, "y": 0.3, "width": 0.25, "height": 0.3},
        )
        self.assertFalse(targets[1].recommended)

    def test_load_edit_targets_marks_only_one_recommended_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "snapshot_payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "canvas": {"widthPx": 200, "heightPx": 100},
                        "visibleObjects": [
                            {
                                "id": "wardrobe",
                                "type": "wardrobe",
                                "screenBboxPx": {
                                    "minX": 20,
                                    "minY": 20,
                                    "width": 40,
                                    "height": 40,
                                },
                            },
                            {
                                "id": "desk",
                                "type": "desk",
                                "screenBboxPx": {
                                    "minX": 100,
                                    "minY": 40,
                                    "width": 40,
                                    "height": 20,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            targets = load_edit_targets(payload_path)

        self.assertEqual(sum(1 for target in targets if target.recommended), 1)
        self.assertEqual(targets[0].object_id, "wardrobe")
        self.assertTrue(targets[0].recommended)

    def test_manual_edit_manifest_loads_and_links_create_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _write_create_manifest(root)
            create_case_dir = (
                root
                / "first_time"
                / "curated_core14_create_batch1"
                / "case_a"
                / "contract_objects_preflight_locked_geometry"
            )
            _write_png(create_case_dir / "render.png")
            _write_passed_metrics(create_case_dir / "metrics.json")
            batches = load_manual_edit_batches(
                manifest_paths=[source_manifest],
                create_runs_root=root / "first_time",
                output_root=root / "edit_manual",
            )
            case_key = batches[0].cases[0].key
            annotation_path = root / "annotations.json"

            result = build_manual_edit_manifests(
                batches=batches,
                annotations={case_key: _annotation()},
                annotation_path=annotation_path,
                manifest_output_prefix=root / "tknt_edit_manual",
            )

            self.assertEqual(len(result.manifest_paths), 1)
            manifest = load_manifest(result.manifest_paths[0])
            self.assertEqual(len(manifest.cases), 1)
            edit_case = manifest.cases[0]
            self.assertEqual(edit_case.id, "case_a__edit_manual")
            self.assertIsNone(edit_case.preset_selection)
            self.assertEqual(edit_case.edit_operations[0].bbox_norm.x, 0.1)
            self.assertEqual(batches[0].cases[0].edit_targets[0].object_id, "sofa")
            linked_case_dir = (
                root / "edit_manual" / "curated_core14_create_batch1" / "case_a"
            )
            self.assertTrue(linked_case_dir.exists())
            self.assertTrue(batches[0].cases[0].guide_path.is_file())

    def test_replace_operation_uses_matching_replacement_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _write_create_manifest(root)
            create_case_dir = (
                root
                / "first_time"
                / "curated_core14_create_batch1"
                / "case_a"
                / "contract_objects_preflight_locked_geometry"
            )
            _write_png(create_case_dir / "render.png")
            _write_passed_metrics(create_case_dir / "metrics.json")
            replacement_dir = root / "replacement_images"
            replacement_image = replacement_dir / "sofa.jpeg"
            _write_png(replacement_image)
            batches = load_manual_edit_batches(
                manifest_paths=[source_manifest],
                create_runs_root=root / "first_time",
                output_root=root / "edit_manual",
            )
            case_key = batches[0].cases[0].key

            result = build_manual_edit_manifests(
                batches=batches,
                annotations={case_key: _replace_annotation()},
                annotation_path=root / "annotations.json",
                manifest_output_prefix=root / "tknt_edit_manual",
                replacement_image_dir=replacement_dir,
            )
            manifest = load_manifest(result.manifest_paths[0])
            operation = manifest.cases[0].edit_operations[0]

        self.assertEqual(operation.operation_type, "replace")
        self.assertEqual(operation.replacement_image, replacement_image.resolve())

    def test_source_approval_overrides_edit_source_variant_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _write_create_manifest(root)
            locked_dir = (
                root
                / "first_time"
                / "curated_core14_create_batch1"
                / "case_a"
                / "contract_objects_preflight_locked_geometry"
            )
            simple_dir = (
                root
                / "first_time"
                / "curated_core14_create_batch1"
                / "case_a"
                / "simple_prompt"
            )
            _write_png(locked_dir / "render.png")
            _write_passed_metrics(locked_dir / "metrics.json")
            _write_png(simple_dir / "render.png")
            approval_path = root / "edit_source_approvals.json"
            approval_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": {
                            "batch1:case_a": {
                                "batch_number": 1,
                                "case_id": "case_a",
                                "variant": "simple_prompt",
                                "render_path": str(simple_dir / "render.png"),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            batches = load_manual_edit_batches(
                manifest_paths=[source_manifest],
                create_runs_root=root / "first_time",
                output_root=root / "edit_manual",
                approval_path=approval_path,
            )
            case = batches[0].cases[0]
            result = build_manual_edit_manifests(
                batches=batches,
                annotations={case.key: _annotation()},
                annotation_path=root / "annotations.json",
                manifest_output_prefix=root / "tknt_edit_manual",
            )
            manifest_payload = json.loads(result.manifest_paths[0].read_text())
            edit_source = manifest_payload["cases"][0]["edit_source_variant"]

            self.assertEqual(case.approved_variant, "simple_prompt")
            self.assertEqual(case.approved_selection_mode, "exact")
            self.assertTrue(case.approved_source_manual)
            self.assertEqual(edit_source["variant"], "simple_prompt")
            self.assertEqual(edit_source["selection_mode"], "exact")


def _annotation() -> ManualEditAnnotation:
    return ManualEditAnnotation(
        bbox_norm={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.25},
        operation_type="recolor",
        object_id="manual_box_1",
        object_name="selected region",
        prompt="Change only this selected object.",
        target_color="muted forest green",
    )


def _replace_annotation() -> ManualEditAnnotation:
    return ManualEditAnnotation(
        bbox_norm={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.25},
        operation_type="replace",
        object_id="sofa",
        object_name="sofa",
        prompt="Replace only this selected sofa.",
        target_color=None,
    )


def _write_create_manifest(root: Path) -> Path:
    case_dir = root / "case_a"
    case_dir.mkdir()
    snapshot_payload = case_dir / "snapshot_payload.json"
    snapshot_payload.write_text(
        json.dumps(
            {
                "canvas": {"widthPx": 100, "heightPx": 100},
                "visibleObjects": [
                    {
                        "id": "sofa",
                        "type": "sofa",
                        "screenBboxPx": {
                            "minX": 20,
                            "minY": 30,
                            "width": 30,
                            "height": 20,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot_image = case_dir / "snapshot.png"
    _write_png(snapshot_image)
    manifest_path = root / "manifest.curated_core14_create.batch1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "metrics": ["blank", "layout_fidelity"],
                    "preset_selection": {
                        "style": "japandi",
                        "lighting": "daylight_6",
                        "scenery": "garden",
                    },
                },
                "output_dir": str(root / "unused"),
                "cases": [
                    {
                        "id": "case_a",
                        "snapshot_payload": str(snapshot_payload),
                        "snapshot_image": str(snapshot_image),
                        "render_prompt": "Photorealize.",
                        "variants": [
                            {
                                "id": "contract_objects_preflight_locked_geometry",
                                "mode": "contract_objects_preflight_locked_geometry",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)


def _write_passed_metrics(path: Path) -> None:
    path.write_text(
        '[{"name": "layout_fidelity", "status": "passed", "value": 0.9}]',
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
