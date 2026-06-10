from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.image_flow_eval.build_stabilization_manifest import (
    _object_bbox_norm,
    build_manifest,
)
from experiments.image_flow_eval.image_flow_eval.io import load_manifest


class StabilizationManifestTest(unittest.TestCase):
    def test_object_bbox_is_derived_from_current_snapshot_payload(self) -> None:
        bbox = _object_bbox_norm(
            {
                "canvas": {"widthPx": 200, "heightPx": 100},
                "visibleObjects": [
                    {
                        "id": "chair",
                        "screenBboxPx": {
                            "minX": 40,
                            "minY": 25,
                            "width": 50,
                            "height": 20,
                        },
                    }
                ],
            },
            "chair",
        )

        self.assertEqual(
            bbox,
            {"x": 0.2, "y": 0.25, "width": 0.25, "height": 0.2},
        )

    def test_compact_manifest_uses_curated_create_variants_for_edits(self) -> None:
        manifest = build_manifest("compact_reading")
        cases = manifest["cases"]
        assert isinstance(cases, list)

        self.assertEqual(len(cases), 5)
        create_case = cases[0]
        edit_case = cases[1]
        assert isinstance(create_case, dict)
        assert isinstance(edit_case, dict)
        defaults = manifest["defaults"]
        assert isinstance(defaults, dict)
        self.assertEqual(
            defaults["metrics"],
            [
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
            ],
        )
        self.assertEqual(
            defaults["preset_selection"],
            {"style": "japandi", "lighting": "daylight_6", "scenery": "garden"},
        )
        variants = create_case["variants"]
        assert isinstance(variants, list)
        self.assertEqual(
            [variant["id"] for variant in variants],
            [
                "contract_objects_preflight_locked_geometry",
                "llm_visual_preflight",
                "llm_visual_preflight_object_anchors",
                "single_edit_edge_silhouette_guided",
                "contract_objects_preflight",
                "contract_objects",
                "simple_prompt",
            ],
        )
        self.assertEqual(
            edit_case["edit_source_variant"],
            {
                "case_id": create_case["id"],
                "variant": "contract_objects_preflight_locked_geometry",
            },
        )

    def test_manifest_parser_accepts_edit_source_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "edit_case",
                                "edit_source_variant": {
                                    "case_id": "create_case",
                                    "variant": "contract_objects_preflight",
                                    "selection_mode": "exact",
                                },
                                "variants": [{"id": "operation_dsl_preflight"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            case = load_manifest(path).cases[0]

        self.assertIsNotNone(case.edit_source_variant)
        assert case.edit_source_variant is not None
        self.assertEqual(case.edit_source_variant.case_id, "create_case")
        self.assertEqual(case.edit_source_variant.variant, "contract_objects_preflight")
        self.assertEqual(case.edit_source_variant.selection_mode, "exact")


if __name__ == "__main__":
    unittest.main()
