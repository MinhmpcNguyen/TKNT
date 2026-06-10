from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.image_flow_eval.prepare_layout_snapshots import (
    BenchmarkCase,
    CameraCandidate,
    LayoutArtifact,
    SkippedLayoutCase,
    _build_layout_payload,
    _camera_presets_for_capture,
    _layout_inventory_integrity_report,
    _load_benchmark_run_cases,
    _load_selected_camera_preset_id,
    _prepare_layout_artifact,
    _score_snapshot_candidate,
    _select_cases,
    _write_camera_snapshot_gallery,
    _write_generated_manifest,
)


class PrepareLayoutSnapshotsTest(unittest.TestCase):
    def test_select_cases_preserves_requested_order(self) -> None:
        cases = [
            _case("living_01_rect_sofa_tv"),
            _case("bedroom_01_rect_door_right_window_left"),
        ]

        selected = _select_cases(
            cases,
            include_case_ids=[
                "bedroom_01_rect_door_right_window_left",
                "living_01_rect_sofa_tv",
            ],
        )

        self.assertEqual(
            [item.case_id for item in selected],
            [
                "bedroom_01_rect_door_right_window_left",
                "living_01_rect_sofa_tv",
            ],
        )

    def test_select_cases_rejects_unknown_case(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_case"):
            _select_cases([_case("known_case")], include_case_ids=["missing_case"])

    def test_score_snapshot_candidate_prefers_useful_mid_corner_view(self) -> None:
        useful_score, useful_details = _score_snapshot_candidate(
            preset={"id": "mid-corner-1", "group": "mid"},
            snapshot_artifact={
                "totalObjectCount": 5,
                "visibleObjectCount": 4,
                "canvas": {"widthPx": 1000, "heightPx": 800},
                "visibilityDiagnostics": {"status": "ok"},
                "visibleObjects": [
                    {
                        "screenBboxPx": {
                            "minX": 220,
                            "minY": 180,
                            "maxX": 420,
                            "maxY": 390,
                        }
                    },
                    {
                        "screenBboxPx": {
                            "minX": 480,
                            "minY": 200,
                            "maxX": 680,
                            "maxY": 420,
                        }
                    },
                    {
                        "screenBboxPx": {
                            "minX": 300,
                            "minY": 430,
                            "maxX": 520,
                            "maxY": 650,
                        }
                    },
                    {
                        "screenBboxPx": {
                            "minX": 600,
                            "minY": 420,
                            "maxX": 760,
                            "maxY": 620,
                        }
                    },
                ],
            },
        )
        empty_score, _ = _score_snapshot_candidate(
            preset={"id": "ceiling-wall-1", "group": "ceiling"},
            snapshot_artifact={
                "totalObjectCount": 5,
                "visibleObjectCount": 0,
                "canvas": {"widthPx": 1000, "heightPx": 800},
                "visibilityDiagnostics": {"status": "degraded"},
                "visibleObjects": [],
            },
        )

        self.assertGreater(useful_score, empty_score)
        self.assertEqual(useful_details["visible_object_count"], 4)
        self.assertEqual(useful_details["preset_group"], "mid")

    def test_write_generated_manifest_uses_relative_snapshot_paths(self) -> None:
        with self.subTest("manifest paths"), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "living_01_rect_sofa_tv"
            case_dir.mkdir()
            (case_dir / "snapshot_payload.json").write_text("{}", encoding="utf-8")
            (case_dir / "snapshot.png").write_bytes(b"png")
            (case_dir / "camera_snapshots").mkdir()
            (case_dir / "camera_snapshots" / "index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            artifact = LayoutArtifact(
                case=_case("living_01_rect_sofa_tv"),
                case_output_dir=case_dir,
                layout_path=case_dir / "layout.json",
                pipeline_output_path=case_dir / "pipeline_output.json",
            )

            manifest_path = _write_generated_manifest(
                output_root=root,
                artifacts=[artifact],
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["cases"][0]["snapshot_payload"],
                "living_01_rect_sofa_tv/snapshot_payload.json",
            )
            self.assertEqual(
                payload["cases"][0]["snapshot_image"],
                "living_01_rect_sofa_tv/snapshot.png",
            )
            self.assertEqual(
                payload["cases"][0]["camera_snapshots"],
                "living_01_rect_sofa_tv/camera_snapshots/index.json",
            )

    def test_load_benchmark_run_cases_discovers_nested_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "tknt_new_s_o" / "simple_case__tknt_full"
            run_dir.mkdir(parents=True)
            (run_dir / "input.json").write_text(
                json.dumps(
                    {
                        "user_input": {
                            "description": "A simple imported benchmark case.",
                            "room_type": "bedroom",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "output.json").write_text(
                json.dumps({"case_id": "simple_case__tknt_full"}),
                encoding="utf-8",
            )

            cases = _load_benchmark_run_cases([str(root / "tknt_new_*")])

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "tknt_new_s_o_simple_case_tknt_full")
            self.assertEqual(cases[0].room_type, "bedroom")
            self.assertEqual(cases[0].raw["benchmark_run_dir"], str(run_dir.resolve()))

    def test_prepare_layout_artifact_skips_missing_regular_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaisesRegex(SkippedLayoutCase, "missing reusable layout"):
                _prepare_layout_artifact(
                    benchmark_case=_case("missing_layout_case"),
                    output_root=root / "cases",
                    layout_cases_root=root / "layout_artifacts",
                    user="image_eval_user",
                    timeout_s=1,
                    reuse_existing=True,
                    skip_missing_layout=True,
                    option_index=1,
                )

    def test_write_camera_snapshot_gallery_writes_every_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case_1"
            case_dir.mkdir()
            artifact = LayoutArtifact(
                case=_case("case_1"),
                case_output_dir=case_dir,
                layout_path=case_dir / "layout.json",
                pipeline_output_path=case_dir / "pipeline_output.json",
            )
            candidates = [
                _candidate("mid-corner-1", score=10.0),
                _candidate("mid-corner-2", score=20.0),
            ]

            index_path = _write_camera_snapshot_gallery(
                artifact=artifact,
                candidates=candidates,
                selected_candidate=candidates[1],
            )

            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["snapshot_count"], 2)
            self.assertEqual(
                len(list((case_dir / "camera_snapshots").glob("*.png"))), 2
            )
            self.assertTrue(payload["snapshots"][1]["selected"])

    def test_write_camera_snapshot_gallery_saves_geometry_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case_1"
            case_dir.mkdir()
            artifact = LayoutArtifact(
                case=_case("case_1"),
                case_output_dir=case_dir,
                layout_path=case_dir / "layout.json",
                pipeline_output_path=case_dir / "pipeline_output.json",
            )
            data_url = _candidate(
                "mid-corner-1", score=10.0
            ).clean_snapshot_image_data_url
            candidate = CameraCandidate(
                preset={"id": "mid-corner-1", "group": "mid"},
                score=10.0,
                details={"visible_object_count": 1},
                snapshot_artifact={"visibleObjectCount": 1},
                clean_snapshot_image_data_url=data_url,
                depth_map_image_data_url=data_url,
                edge_map_image_data_url=data_url,
                segmentation_map_image_data_url=data_url,
                instance_outline_image_data_url=data_url,
                instance_outline_overlay_image_data_url=data_url,
            )

            index_path = _write_camera_snapshot_gallery(
                artifact=artifact,
                candidates=[candidate],
                selected_candidate=candidate,
            )

            camera_dir = case_dir / "camera_snapshots"
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            geometry_maps = payload["snapshots"][0]["geometry_maps"]
            self.assertTrue((camera_dir / geometry_maps["depth_map"]).is_file())
            self.assertTrue((camera_dir / geometry_maps["edge_map"]).is_file())
            self.assertTrue((camera_dir / geometry_maps["segmentation_map"]).is_file())
            self.assertTrue(
                (camera_dir / geometry_maps["instance_outline_guide"]).is_file()
            )
            self.assertTrue(
                (camera_dir / geometry_maps["instance_outline_overlay"]).is_file()
            )

    def test_load_selected_camera_preset_id_reads_existing_meta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snapshot_meta.json").write_text(
                json.dumps({"selected_preset": {"id": "mid-corner-3"}}),
                encoding="utf-8",
            )

            self.assertEqual(
                _load_selected_camera_preset_id(root),
                "mid-corner-3",
            )

    def test_camera_presets_for_capture_reuses_selected_preset_only(self) -> None:
        presets = [
            {"id": "mid-corner-1"},
            {"id": "mid-corner-2"},
            {"id": "mid-corner-3"},
        ]

        selected = _camera_presets_for_capture(
            presets=presets,
            selected_preset_id="mid-corner-2",
            max_camera_presets=1,
        )

        self.assertEqual(selected, [{"id": "mid-corner-2"}])

    def test_build_layout_payload_uses_explicit_module_outputs_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale_dir = root / "stale"
            fresh_dir = root / "fresh"
            (stale_dir / "module_outputs").mkdir(parents=True)
            fresh_dir.mkdir()
            _write_layout_variants(
                stale_dir / "module_outputs" / "layout_variants.json",
                ["sofa", "armchair_2"],
            )
            _write_layout_variants(
                fresh_dir / "layout_variants.json",
                ["sofa"],
            )

            payload = _build_layout_payload(
                benchmark_case=_case("case_1"),
                pipeline_payload={"case_dir": str(stale_dir)},
                option_index=1,
                module_outputs_dir=fresh_dir,
            )

        styled_objects = payload["styled_result"]["objects"]
        self.assertEqual(
            [item["instance_id"] for item in styled_objects],
            ["sofa"],
        )

    def test_layout_inventory_integrity_report_rejects_extra_object(self) -> None:
        report = _layout_inventory_integrity_report(
            pipeline_payload={
                "final_output": {"objects": [{"instance_id": "sofa"}]},
            },
            layout_payload={
                "styled_result": {
                    "objects": [
                        {"instance_id": "sofa"},
                        {"instance_id": "armchair_2"},
                    ]
                },
                "design": {
                    "placements": [
                        {"placement_id": "sofa"},
                        {"placement_id": "armchair_2"},
                    ]
                },
            },
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["extra_styled_ids"], ["armchair_2"])


def _case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        room_type="living_room",
        input_payload={
            "user_input": {
                "description": f"Description for {case_id}",
                "room_type": "living_room",
                "shape_points": [
                    {"x": 0, "y": 0},
                    {"x": 1000, "y": 0},
                    {"x": 1000, "y": 1000},
                    {"x": 0, "y": 1000},
                ],
            }
        },
        raw={"case_id": case_id},
    )


def _write_layout_variants(path: Path, object_ids: list[str]) -> None:
    objects = [
        {
            "instance_id": object_id,
            "object_type": object_id,
            "bbox": {"min_x": 0, "min_y": 0, "max_x": 100, "max_y": 100},
        }
        for object_id in object_ids
    ]
    path.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "variant_id": "variant_1",
                        "styled_result": {"objects": objects},
                        "absolute_layout": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _candidate(preset_id: str, *, score: float) -> CameraCandidate:
    return CameraCandidate(
        preset={"id": preset_id, "group": "mid"},
        score=score,
        details={"visible_object_count": 1},
        snapshot_artifact={"visibleObjectCount": 1},
        clean_snapshot_image_data_url=(
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+"
            "M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ),
    )


if __name__ == "__main__":
    unittest.main()
