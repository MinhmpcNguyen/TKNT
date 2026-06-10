from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.image_flow_eval.select_edit_source_variants import (
    EditSourceSelectorApp,
    load_edit_source_approvals,
    load_selection_cases,
    load_variant_candidates,
)


class EditSourceSelectorToolTest(unittest.TestCase):
    def test_load_variant_candidates_scores_from_structural_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant_dir = root / "case_a" / "simple_prompt"
            _write_png(variant_dir / "render.png")
            _write_png(variant_dir / "source_3d.png")
            _write_metrics(
                variant_dir / "metrics.json",
                [
                    {
                        "name": "structural_layout_score",
                        "status": "passed",
                        "value": 0.72,
                        "threshold": 0.58,
                    },
                    {
                        "name": "layout_gate_score",
                        "status": "failed",
                        "value": 0.31,
                        "threshold": 0.65,
                    },
                    {
                        "name": "layout_fidelity",
                        "status": "failed",
                        "value": 0.3,
                        "details": {
                            "reason": "A new door appears on the left wall.",
                            "hard_check_failures": ["openings"],
                        },
                    },
                ],
            )
            ignored_variant_dir = root / "case_a" / "llm_visual_preflight_object_anchors"
            _write_png(ignored_variant_dir / "render.png")
            _write_metrics(
                ignored_variant_dir / "metrics.json",
                [
                    {
                        "name": "layout_gate_score",
                        "status": "passed",
                        "value": 0.99,
                    }
                ],
            )

            candidates = load_variant_candidates(root / "case_a")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].variant, "simple_prompt")
        self.assertEqual(candidates[0].score_metric, "structural_layout_score")
        self.assertEqual(candidates[0].score, 0.72)
        self.assertEqual(candidates[0].score_status, "passed")
        self.assertIn("openings", candidates[0].hard_check_failures)
        self.assertEqual(candidates[0].reason, "A new door appears on the left wall.")

    def test_app_saves_manual_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_manifest(root)
            create_case_dir = (
                root
                / "first_time"
                / "curated_core14_create_batch1"
                / "case_a"
                / "simple_prompt"
            )
            _write_png(create_case_dir / "render.png")
            _write_metrics(
                create_case_dir / "metrics.json",
                [
                    {
                        "name": "layout_gate_score",
                        "status": "passed",
                        "value": 0.8,
                    }
                ],
            )
            approval_path = root / "approvals.json"
            cases = load_selection_cases(
                manifest_paths=[manifest_path],
                create_runs_root=root / "first_time",
            )
            app = EditSourceSelectorApp(cases=cases, approval_path=approval_path)

            app.save_approval(key="batch1:case_a", variant="simple_prompt")
            approvals = load_edit_source_approvals(approval_path)

        self.assertEqual(approvals["batch1:case_a"].variant, "simple_prompt")
        self.assertEqual(approvals["batch1:case_a"].score, 0.8)

    def test_load_selection_cases_keeps_manifest_case_count_while_filtering_variants(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_manifest(root, case_ids=("case_a", "case_b"))
            for case_id in ("case_a", "case_b"):
                create_case_dir = (
                    root
                    / "first_time"
                    / "curated_core14_create_batch1"
                    / case_id
                )
                _write_png(create_case_dir / "simple_prompt" / "render.png")
                _write_png(
                    create_case_dir
                    / "llm_visual_preflight_object_anchors"
                    / "render.png"
                )

            cases = load_selection_cases(
                manifest_paths=[manifest_path],
                create_runs_root=root / "first_time",
                variant_ids=("simple_prompt",),
            )

        self.assertEqual(len(cases), 2)
        self.assertEqual(
            [candidate.variant for candidate in cases[0].candidates],
            ["simple_prompt"],
        )
        self.assertEqual(
            [candidate.variant for candidate in cases[1].candidates],
            ["simple_prompt"],
        )


def _write_manifest(root: Path, *, case_ids: tuple[str, ...] = ("case_a",)) -> Path:
    cases: list[dict[str, object]] = []
    for case_id in case_ids:
        case_dir = root / case_id
        case_dir.mkdir()
        snapshot = case_dir / "snapshot.png"
        payload = case_dir / "snapshot_payload.json"
        _write_png(snapshot)
        payload.write_text("{}", encoding="utf-8")
        cases.append(
            {
                "id": case_id,
                "snapshot_image": str(snapshot),
                "snapshot_payload": str(payload),
                "variants": [
                    {
                        "id": "simple_prompt",
                        "mode": "simple_prompt",
                    }
                ],
            }
        )
    manifest_path = root / "manifest.curated_core14_create.batch1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)


def _write_metrics(path: Path, metrics: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(metrics), encoding="utf-8")
