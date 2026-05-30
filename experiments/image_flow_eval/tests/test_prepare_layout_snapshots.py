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
    _load_benchmark_run_cases,
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


def _case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        room_type="living_room",
        input_payload={
            "user_input": {
                "description": f"Description for {case_id}",
                "room_type": "living_room",
            }
        },
        raw={"case_id": case_id},
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
