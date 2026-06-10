from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.image_flow_eval.build_curated_create_manifests import (
    _CORE_CASE_IDS,
    _NO_SCENERY_CASE_IDS,
    _PRESET_FOLLOW_CASE_IDS,
    _PRESET_PACKS,
    build_core_batch_manifests,
    build_core_manifest,
    build_preset_follow_manifest,
    write_curated_manifests,
)
from experiments.image_flow_eval.image_flow_eval.io import load_manifest


class CuratedCreateManifestsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.selected_cases_path = self.root / "selected_snapshot_cases.json"
        selected_cases = []
        for case_id in [
            *_CORE_CASE_IDS,
            "tknt_new_hard_nonrect_proof_living_l_no_openings_tknt_full",
        ]:
            case_dir = self.root / case_id
            case_dir.mkdir()
            snapshot_image = case_dir / "snapshot.png"
            snapshot_image.touch()
            snapshot_payload = case_dir / "snapshot_payload.json"
            snapshot_payload.write_text("{}", encoding="utf-8")
            selected_cases.append(
                {
                    "case_id": case_id,
                    "snapshot_image": str(snapshot_image),
                    "snapshot_payload": str(snapshot_payload),
                }
            )
        self.selected_cases_path.write_text(
            json.dumps(
                {
                    "primary_existing_cases": selected_cases,
                    "primary_hard_cases": [],
                    "secondary_hard_cases": [],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_core_manifest_contains_requested_fourteen_cases(self) -> None:
        manifest = build_core_manifest(
            self.selected_cases_path,
            output_dir=self.root / "runs" / "core",
        )
        cases = manifest["cases"]
        assert isinstance(cases, list)

        self.assertEqual(
            [case["id"] for case in cases if isinstance(case, dict)],
            list(_CORE_CASE_IDS),
        )
        self.assertNotIn(
            "tknt_new_hard_nonrect_proof_living_l_no_openings_tknt_full",
            [case["id"] for case in cases if isinstance(case, dict)],
        )
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
        thresholds = defaults["thresholds"]
        assert isinstance(thresholds, dict)
        self.assertEqual(thresholds["min_layout_gate_score"], 0.65)
        self.assertEqual(thresholds["min_layout_edge_structure_score"], 0.35)
        self.assertEqual(thresholds["min_layout_depth_structure_score"], 0.45)
        self.assertEqual(thresholds["min_layout_dino_patch_similarity"], 0.50)
        self.assertEqual(thresholds["min_structural_layout_score"], 0.58)

    def test_core_batches_keep_three_cases_per_run(self) -> None:
        manifests = build_core_batch_manifests(
            self.selected_cases_path,
            runs_root=self.root / "runs",
        )

        self.assertEqual(
            [len(manifest["cases"]) for _, manifest in manifests], [3, 3, 3, 3, 2]
        )

    def test_core_manifest_removes_scenery_for_no_window_cases(self) -> None:
        manifest = build_core_manifest(
            self.selected_cases_path,
            output_dir=self.root / "runs" / "core",
        )
        cases = manifest["cases"]
        assert isinstance(cases, list)
        cases_by_id = {
            case["id"]: case
            for case in cases
            if isinstance(case, dict) and isinstance(case.get("id"), str)
        }

        for case_id in _NO_SCENERY_CASE_IDS:
            case = cases_by_id[case_id]
            preset_selection = case["preset_selection"]
            assert isinstance(preset_selection, dict)
            self.assertEqual(preset_selection["style"], "japandi")
            self.assertEqual(preset_selection["lighting"], "daylight_6")
            self.assertNotIn("scenery", preset_selection)

        default_preset = manifest["defaults"]["preset_selection"]
        assert isinstance(default_preset, dict)
        self.assertEqual(default_preset["scenery"], "garden")

    def test_preset_follow_manifest_crosses_four_cases_with_three_presets(self) -> None:
        manifest = build_preset_follow_manifest(
            self.selected_cases_path,
            output_dir=self.root / "runs" / "preset-follow",
        )
        cases = manifest["cases"]
        assert isinstance(cases, list)

        self.assertEqual(len(cases), len(_PRESET_FOLLOW_CASE_IDS) * len(_PRESET_PACKS))
        for case in cases:
            assert isinstance(case, dict)
            self.assertIn("preset_selection", case)

    def test_preset_follow_manifest_removes_scenery_for_no_window_cases(self) -> None:
        manifest = build_preset_follow_manifest(
            self.selected_cases_path,
            output_dir=self.root / "runs" / "preset-follow",
        )
        cases = manifest["cases"]
        assert isinstance(cases, list)
        no_scenery_cases = [
            case
            for case in cases
            if isinstance(case, dict)
            and str(case.get("id", "")).startswith(
                "tknt_new_hard_nonrect_proof_bedroom_l_openings_tknt_full"
            )
        ]

        self.assertEqual(len(no_scenery_cases), len(_PRESET_PACKS))
        for case in no_scenery_cases:
            preset_selection = case["preset_selection"]
            assert isinstance(preset_selection, dict)
            self.assertIn("style", preset_selection)
            self.assertIn("lighting", preset_selection)
            self.assertNotIn("scenery", preset_selection)

    def test_written_manifests_are_accepted_by_parser(self) -> None:
        written_paths = write_curated_manifests(
            self.selected_cases_path,
            output_root=self.root / "audit",
            runs_root=self.root / "runs",
        )

        self.assertEqual(len(written_paths), 10)
        for path in written_paths:
            load_manifest(path)


if __name__ == "__main__":
    unittest.main()
