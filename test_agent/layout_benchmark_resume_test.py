from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.layout_benchmark.run_benchmark import (  # noqa: E402
    BENCHMARK_DIR,
    _build_plan,
    _load_systems,
    _run_plan,
    _resume_output_complete,
    _write_ablation_summary,
)


class LayoutBenchmarkResumeTest(unittest.TestCase):
    def test_resume_skips_readable_output_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"
            output_path.write_text('{"objects": []}\n', encoding="utf-8")

            self.assertTrue(_resume_output_complete({"output_path": str(output_path)}))

    def test_resume_reruns_missing_output_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing.json"

            self.assertFalse(_resume_output_complete({"output_path": str(output_path)}))

    def test_resume_reruns_malformed_output_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"
            output_path.write_text("{bad json", encoding="utf-8")

            self.assertFalse(_resume_output_complete({"output_path": str(output_path)}))

    def test_resume_run_plan_skips_existing_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            output_path = output_root / "runs" / "case__system" / "output.json"
            output_path.parent.mkdir(parents=True)
            output_path.write_text('{"objects": []}\n', encoding="utf-8")

            _run_plan(
                plan_rows=[
                    {
                        "run_id": "case__system",
                        "case_id": "case",
                        "system_id": "system",
                        "output_path": str(output_path),
                    }
                ],
                cases=[],
                systems=[],
                output_root=output_root,
                external_root=output_root / "external",
                cases_root=output_root / "cases",
                user="benchmark_user",
                timeout_s=1,
                external_timeout_s=1,
                max_runs=0,
                run_external=True,
                model="gpt-5.4-mini",
                idesign_objects=8,
                external_max_attempts=1,
                resume=True,
            )

            report = json.loads(
                (output_root / "resume_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["skipped_count"], 1)
            self.assertEqual(report["processed_count"], 0)

    def test_ablation_systems_manifest_loads_clean_component_modes(self) -> None:
        systems = _load_systems(BENCHMARK_DIR / "systems_ablation.json")
        modes = {system.ablation for system in systems}

        self.assertIn("full", modes)
        self.assertIn("minimal_room_context", modes)
        self.assertIn("no_stylist_llm", modes)
        self.assertIn("no_gallery_selector", modes)
        self.assertNotIn("no_capacity_control", modes)

    def test_ablation_run_plan_includes_every_system(self) -> None:
        systems = _load_systems(BENCHMARK_DIR / "systems_ablation.json")
        with TemporaryDirectory() as temp_dir:
            rows = _build_plan(
                cases=[{"case_id": "case_1"}],
                systems=systems,
                output_root=Path(temp_dir),
                model="gpt-5.4-mini",
            )

        self.assertEqual(len(rows), len(systems))
        self.assertEqual(
            {row["system_id"] for row in rows},
            {system.system_id for system in systems},
        )

    def test_ablation_summary_writes_deltas_against_tknt_full(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)

            _write_ablation_summary(
                output_root,
                [
                    {
                        "system_id": "tknt_full",
                        "overall_score": 80.0,
                        "variant_evaluation": {
                            "canonical_overall_score": 70.0,
                            "best_overall_score": 80.0,
                            "variant_count": 5,
                        },
                        "component_scores": {
                            "geometry": 90.0,
                            "request_following": 75.0,
                            "arrangement_reasonableness": 80.0,
                            "valid_object_count": 75.0,
                        },
                        "metrics": {
                            "objects_out_of_bounds": 1,
                            "overlapping_blocking_pairs": 2,
                            "valid_blocking_object_count": 6,
                        },
                    },
                    {
                        "system_id": "tknt_ab_no_stylist_llm",
                        "overall_score": 72.0,
                        "variant_evaluation": {
                            "canonical_overall_score": 68.0,
                            "best_overall_score": 72.0,
                            "variant_count": 5,
                        },
                        "component_scores": {
                            "geometry": 85.0,
                            "request_following": 70.0,
                            "arrangement_reasonableness": 72.0,
                            "valid_object_count": 60.0,
                        },
                        "metrics": {
                            "objects_out_of_bounds": 0,
                            "overlapping_blocking_pairs": 1,
                            "valid_blocking_object_count": 5,
                        },
                    },
                ],
            )

            summary = json.loads(
                (output_root / "ablation_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue((output_root / "ablation_summary.csv").is_file())

        by_system = {row["system_id"]: row for row in summary}
        ablated = by_system["tknt_ab_no_stylist_llm"]
        self.assertEqual(ablated["delta_vs_tknt_full_overall"], -8.0)
        self.assertEqual(ablated["delta_vs_tknt_full_canonical"], -2.0)
        self.assertEqual(ablated["mean_best_gallery_overall_score"], 72.0)


if __name__ == "__main__":
    unittest.main()
