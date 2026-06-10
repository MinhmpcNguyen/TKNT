from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments.image_flow_eval.image_flow_eval.io import encode_image_to_data_url
from experiments.image_flow_eval.image_flow_eval.semantic_repair import (
    SemanticRepairInput,
    discover_semantic_repair_inputs,
    run_semantic_repair_case,
)


class SemanticRepairTest(unittest.TestCase):
    def test_discovers_reusable_semantic_pass_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_semantic_case(root / "run", "case_a")

            inputs = discover_semantic_repair_inputs(
                run_dir=root / "run",
                output_dir=root / "repair",
            )

            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0].case_id, "case_a")
            self.assertEqual(
                inputs[0].output_variant_dir.name,
                "single_edit_edge_silhouette_guided_llm_repair",
            )

    def test_repair_uses_current_render_as_edit_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair_input = _make_semantic_case(root / "run", "case_a")
            output_dir = root / "repair" / "case_a" / "semantic_repair"
            repair_input = SemanticRepairInput(
                case_id=repair_input.case_id,
                source_variant_dir=repair_input.source_variant_dir,
                output_variant_dir=output_dir,
                source_image=repair_input.source_image,
                pass1_render_image=repair_input.pass1_render_image,
                source_payload=repair_input.source_payload,
                semantic_guide_image=repair_input.semantic_guide_image,
            )
            output_dir.mkdir(parents=True)
            _write_plan(
                output_dir / "llm_repair_plan.json",
                repair_needed=True,
                prompt="Restore the left wall plane and keep the TV cluster unchanged.",
            )
            mock_result = {
                "image": {
                    "data_url": encode_image_to_data_url(
                        repair_input.pass1_render_image
                    )
                }
            }

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value=mock_result,
            ) as renderer:
                result = run_semantic_repair_case(repair_input, resume=True)

            self.assertEqual(result.status, "generated")
            self.assertTrue((output_dir / "render.png").is_file())
            kwargs = renderer.call_args.kwargs
            self.assertEqual(kwargs["render_mode"], "edit")
            self.assertEqual(
                kwargs["scene_reference_mode"],
                "scene_reference_camera_transfer",
            )
            self.assertEqual(
                kwargs["edit_source_image_data_url"],
                encode_image_to_data_url(output_dir / "semantic_pass_render.png"),
            )
            self.assertEqual(
                kwargs["snapshot_image_data_url"],
                encode_image_to_data_url(output_dir / "source_3d.png"),
            )
            self.assertIn("left wall", kwargs["user_prompt"])
            config = kwargs["config"]
            self.assertFalse(config.include_preset_reference_images)
            self.assertTrue(config.include_annotated_reference)

    def test_enhanced_geometry_guide_is_used_as_annotated_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair_input = _make_semantic_case(root / "run", "case_a")
            output_dir = root / "repair" / "case_a" / "semantic_repair"
            repair_input = SemanticRepairInput(
                case_id=repair_input.case_id,
                source_variant_dir=repair_input.source_variant_dir,
                output_variant_dir=output_dir,
                source_image=repair_input.source_image,
                pass1_render_image=repair_input.pass1_render_image,
                source_payload=repair_input.source_payload,
                semantic_guide_image=repair_input.semantic_guide_image,
            )
            output_dir.mkdir(parents=True)
            _write_plan(
                output_dir / "llm_repair_plan.json",
                repair_needed=True,
                prompt="Restore object scale.",
            )
            mock_result = {
                "image": {
                    "data_url": encode_image_to_data_url(
                        repair_input.pass1_render_image
                    )
                }
            }

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value=mock_result,
            ) as renderer:
                run_semantic_repair_case(
                    repair_input,
                    resume=True,
                    enhanced_geometry_guide=True,
                )

            guide_path = output_dir / "enhanced_geometry_guide.png"
            self.assertTrue(guide_path.is_file())
            kwargs = renderer.call_args.kwargs
            self.assertEqual(
                kwargs["annotated_reference_image_data_url"],
                encode_image_to_data_url(guide_path),
            )

    def test_guide_only_builds_enhanced_geometry_guide_without_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair_input = _make_semantic_case(root / "run", "case_a")
            output_dir = root / "repair" / "case_a" / "semantic_repair"
            repair_input = SemanticRepairInput(
                case_id=repair_input.case_id,
                source_variant_dir=repair_input.source_variant_dir,
                output_variant_dir=output_dir,
                source_image=repair_input.source_image,
                pass1_render_image=repair_input.pass1_render_image,
                source_payload=repair_input.source_payload,
                semantic_guide_image=repair_input.semantic_guide_image,
            )

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path"
            ) as renderer:
                result = run_semantic_repair_case(
                    repair_input,
                    enhanced_geometry_guide=True,
                    guide_only=True,
                )

            self.assertEqual(result.status, "guide_only")
            renderer.assert_not_called()
            self.assertTrue((output_dir / "enhanced_geometry_guide.png").is_file())
            self.assertTrue((output_dir / "tknt_snapshot_result.json").is_file())

    def test_no_repair_needed_copies_semantic_pass_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair_input = _make_semantic_case(root / "run", "case_a")
            output_dir = root / "repair" / "case_a" / "semantic_repair"
            repair_input = SemanticRepairInput(
                case_id=repair_input.case_id,
                source_variant_dir=repair_input.source_variant_dir,
                output_variant_dir=output_dir,
                source_image=repair_input.source_image,
                pass1_render_image=repair_input.pass1_render_image,
                source_payload=repair_input.source_payload,
                semantic_guide_image=repair_input.semantic_guide_image,
            )
            output_dir.mkdir(parents=True)
            _write_plan(output_dir / "llm_repair_plan.json", repair_needed=False)

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path"
            ) as renderer:
                result = run_semantic_repair_case(repair_input, resume=True)

            self.assertEqual(result.status, "skipped_no_repair_needed")
            renderer.assert_not_called()
            self.assertEqual(
                encode_image_to_data_url(output_dir / "render.png"),
                encode_image_to_data_url(repair_input.pass1_render_image),
            )

    def test_force_repair_runs_even_when_plan_says_no_repair_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair_input = _make_semantic_case(root / "run", "case_a")
            output_dir = root / "repair" / "case_a" / "semantic_repair"
            repair_input = SemanticRepairInput(
                case_id=repair_input.case_id,
                source_variant_dir=repair_input.source_variant_dir,
                output_variant_dir=output_dir,
                source_image=repair_input.source_image,
                pass1_render_image=repair_input.pass1_render_image,
                source_payload=repair_input.source_payload,
                semantic_guide_image=repair_input.semantic_guide_image,
            )
            output_dir.mkdir(parents=True)
            _write_plan(output_dir / "llm_repair_plan.json", repair_needed=False)
            mock_result = {
                "image": {
                    "data_url": encode_image_to_data_url(repair_input.source_image)
                }
            }

            with patch(
                "pipeline.snapshot_image_renderer.render_snapshot_image_from_path",
                return_value=mock_result,
            ) as renderer:
                result = run_semantic_repair_case(
                    repair_input,
                    resume=True,
                    force_repair=True,
                )

            self.assertEqual(result.status, "generated")
            self.assertTrue(renderer.called)
            kwargs = renderer.call_args.kwargs
            self.assertIn("Conservatively compare", kwargs["user_prompt"])


def _make_semantic_case(root: Path, case_id: str) -> SemanticRepairInput:
    variant_dir = root / case_id / "single_edit_edge_silhouette_guided"
    variant_dir.mkdir(parents=True)
    source_image = variant_dir / "source_3d.png"
    render_image = variant_dir / "render.png"
    guide_image = variant_dir / "room_shell_semantic_guide.png"
    source_payload = variant_dir / "source_snapshot_payload.json"
    _write_png(source_image, (20, 80, 120, 255))
    _write_png(render_image, (180, 140, 100, 255))
    _write_png(guide_image, (20, 200, 140, 255))
    source_payload.write_text(
        json.dumps(
            {
                "camera": {"presetId": "mid_corner"},
                "openings": {"doors": [], "windows": []},
                "visibleObjects": [
                    {
                        "id": "sofa",
                        "type": "sofa",
                        "screenBboxPx": {
                            "minX": 10,
                            "minY": 20,
                            "width": 30,
                            "height": 40,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return SemanticRepairInput(
        case_id=case_id,
        source_variant_dir=variant_dir,
        output_variant_dir=root.parent / "repair" / case_id / "semantic_repair",
        source_image=source_image,
        pass1_render_image=render_image,
        source_payload=source_payload,
        semantic_guide_image=guide_image,
    )


def _write_png(path: Path, color: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (8, 8), color)
    image.save(path)


def _write_plan(
    path: Path,
    *,
    repair_needed: bool,
    prompt: str = "",
) -> None:
    payload = {
        "repair_needed": repair_needed,
        "overall_status": "needs_repair" if repair_needed else "no_repair_needed",
        "issues": [
            {
                "category": "room_shell",
                "severity": "major",
                "source_evidence": "Image 1 has a left wall.",
                "render_problem": "Image 2 misses the left wall.",
                "repair_instruction": prompt or "No repair needed.",
            }
        ]
        if repair_needed
        else [],
        "repair_prompt": prompt,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
