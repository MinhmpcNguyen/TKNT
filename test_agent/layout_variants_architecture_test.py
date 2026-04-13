from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_cluster_placer = types.ModuleType("agent.cluster_placer")
fake_cluster_placer.Stylist = object
fake_cluster_placer.build_phase2_payload = lambda *args, **kwargs: {}
fake_cluster_placer.compile_phase2_to_final_output = lambda payload: {}
fake_cluster_placer.make_no_improvement_repair = lambda *args, **kwargs: {}
sys.modules.setdefault("agent.cluster_placer", fake_cluster_placer)

fake_stylist = types.ModuleType("agent.stylist")


class _DummyStylist:
    pass


fake_stylist.Stylist = _DummyStylist
fake_stylist.stylist_model_name_for_variant = lambda variant_index: (
    "gemma-4-31b-it",
    "gemini-3.1-flash-lite-preview",
    "gemma-3-27b-it",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",
)[(max(1, int(variant_index)) - 1) % 5]
sys.modules.setdefault("agent.stylist", fake_stylist)

fake_tools_v2 = types.ModuleType("cluster_placer.tools_v2")
fake_tools_v2.EnumeratePhase2RepairMoves = lambda *args, **kwargs: {"moves": []}
fake_tools_v2.EvaluatePhase2Proposal = lambda *args, **kwargs: {
    "hard_valid": True,
    "baseline_comparison": {"candidate_score": 0},
}
fake_tools_v2.PromotePhase2RepairToSeedPayload = lambda *args, **kwargs: {}
sys.modules.setdefault("cluster_placer.tools_v2", fake_tools_v2)

fake_manual = types.ModuleType("pipeline.manual_placements")
fake_manual.merge_manual_placements_into_absolute_layout = (
    lambda layout, *_args, **_kwargs: layout
)
fake_manual.merge_manual_placements_into_styled_output = (
    lambda layout, *_args, **_kwargs: layout
)
sys.modules.setdefault("pipeline.manual_placements", fake_manual)

layout_variants = importlib.import_module("pipeline.layout_variants")


def _layout(cluster_positions: dict[str, tuple[int, int]]) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    for cluster_id, (x_pos, y_pos) in cluster_positions.items():
        objects.append(
            {
                "cluster_id": cluster_id,
                "object_id": f"{cluster_id}_object",
                "instance_id": f"{cluster_id}_object",
                "rotation_ccw": 0,
                "bbox": {
                    "min_x": x_pos,
                    "min_y": y_pos,
                    "max_x": x_pos + 300,
                    "max_y": y_pos + 300,
                },
            }
        )
    return {"status": "OK", "objects": objects}


def _candidate(
    *,
    source: str,
    reason: str,
    score: int,
    layout: dict[str, object],
) -> object:
    return layout_variants._VariantCandidate(
        absolute_layout=layout,
        styled_result=None,
        source=source,
        reason=reason,
        layout_score=score,
        hard_valid=True,
        notes=[],
        promoted_payload={},
        family=layout_variants._variant_family(source, reason),
    )


class LayoutVariantsArchitectureTest(unittest.TestCase):
    def test_select_top_variant_candidates_prefers_complete_gallery_eligible_layouts(
        self,
    ) -> None:
        partial_layout = _layout({"media": (900, 0)})
        partial_layout["complete"] = False
        partial_layout["gallery_eligible"] = False
        partial_layout["coverage_ratio"] = 0.5
        partial_layout["missing_cluster_ids"] = ["living"]

        complete_layout = _layout({"living": (0, 0), "media": (900, 0)})
        complete_layout["complete"] = True
        complete_layout["gallery_eligible"] = True
        complete_layout["coverage_ratio"] = 1.0
        complete_layout["missing_cluster_ids"] = []

        selected = layout_variants.select_top_variant_candidates(
            candidates=[
                {
                    "absolute_layout": partial_layout,
                    "source": "intent_branch:intent_1",
                    "reason": "Partial candidate.",
                    "layout_score": -100,
                    "hard_valid": True,
                    "notes": [],
                    "family": "intent_branch:intent_1",
                },
                {
                    "absolute_layout": complete_layout,
                    "source": "intent_branch:intent_2",
                    "reason": "Complete candidate.",
                    "layout_score": -400,
                    "hard_valid": True,
                    "notes": [],
                    "family": "intent_branch:intent_2",
                },
            ],
            max_variants=2,
        )

        self.assertEqual(selected[0]["source"], "intent_branch:intent_2")

    def test_select_distinct_final_gallery_candidates_backfills_after_distinct_pass(
        self,
    ) -> None:
        room = {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        }
        same_layout = _layout({"living": (0, 0), "media": (900, 0)})
        same_macro_layout = _layout({"living": (60, 0), "media": (960, 0)})
        distinct_macro_layout = _layout({"living": (0, 2200), "media": (900, 2200)})
        same_layout["room"] = room
        same_macro_layout["room"] = room
        distinct_macro_layout["room"] = room

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[
                {
                    "absolute_layout": same_layout,
                    "styled_result": {"objects": []},
                    "source": "seed_stable",
                    "reason": "Stable seed.",
                    "layout_score": 100,
                    "hard_valid": True,
                    "complete": True,
                    "gallery_eligible": True,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
                {
                    "absolute_layout": same_layout,
                    "styled_result": {"objects": []},
                    "source": "seed_duplicate",
                    "reason": "Same state signature.",
                    "layout_score": 90,
                    "hard_valid": True,
                    "complete": True,
                    "gallery_eligible": True,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
                {
                    "absolute_layout": same_macro_layout,
                    "styled_result": {"objects": []},
                    "source": "seed_neighbor",
                    "reason": "Near duplicate macro layout.",
                    "layout_score": 95,
                    "hard_valid": True,
                    "complete": True,
                    "gallery_eligible": True,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
                {
                    "absolute_layout": distinct_macro_layout,
                    "styled_result": {"objects": []},
                    "source": "seed_distinct",
                    "reason": "Different macro layout.",
                    "layout_score": 80,
                    "hard_valid": True,
                    "complete": True,
                    "gallery_eligible": True,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
            ],
            max_variants=5,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0]["source"], "seed_stable")
        self.assertEqual(selected[1]["source"], "seed_distinct")
        self.assertEqual(selected[0]["gallery_selection_mode"], "distinct")
        self.assertEqual(selected[1]["gallery_selection_mode"], "distinct")
        self.assertEqual(selected[2]["gallery_selection_mode"], "backfill")
        self.assertEqual(selected[3]["gallery_selection_mode"], "backfill")
        self.assertIn("state_signature", selected[0])
        self.assertIn("macro_layout_signature", selected[0])

    def test_final_gallery_dedupe_ignores_reintroduced_objects_for_core_layouts(
        self,
    ) -> None:
        room = {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        }
        top_layout = {
            "status": "OK",
            "room": room,
            "objects": [
                {
                    "cluster_id": "seating_zone",
                    "object_id": "sectional_sofa",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 300, "min_y": 200, "max_x": 1300, "max_y": 900},
                },
                {
                    "cluster_id": "seating_zone",
                    "object_id": "ottoman__reintroduced_1",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 200, "min_y": 100, "max_x": 2600, "max_y": 3200},
                },
                {
                    "cluster_id": "media_zone",
                    "object_id": "tv_console",
                    "rotation_ccw": 180,
                    "bbox": {
                        "min_x": 4200,
                        "min_y": 1200,
                        "max_x": 5400,
                        "max_y": 1700,
                    },
                },
            ],
        }
        bottom_layout = {
            "status": "OK",
            "room": room,
            "objects": [
                {
                    "cluster_id": "seating_zone",
                    "object_id": "sectional_sofa",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 300, "min_y": 2300, "max_x": 1300, "max_y": 3000},
                },
                {
                    "cluster_id": "seating_zone",
                    "object_id": "ottoman__reintroduced_1",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 200, "min_y": 100, "max_x": 2600, "max_y": 3200},
                },
                {
                    "cluster_id": "media_zone",
                    "object_id": "tv_console",
                    "rotation_ccw": 180,
                    "bbox": {
                        "min_x": 4200,
                        "min_y": 1200,
                        "max_x": 5400,
                        "max_y": 1700,
                    },
                },
            ],
        }

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[
                {
                    "absolute_layout": top_layout,
                    "styled_result": {"objects": []},
                    "source": "seed_top",
                    "reason": "Top seating.",
                    "layout_score": 100,
                    "hard_valid": True,
                    "complete": True,
                    "gallery_eligible": True,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
                {
                    "absolute_layout": bottom_layout,
                    "styled_result": {"objects": []},
                    "source": "seed_bottom",
                    "reason": "Bottom seating.",
                    "layout_score": 95,
                    "hard_valid": True,
                    "complete": True,
                    "gallery_eligible": True,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
            ],
            max_variants=5,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["source"], "seed_top")
        self.assertEqual(selected[1]["source"], "seed_bottom")

    def test_near_duplicate_detection_uses_object_identity_over_cluster_name(
        self,
    ) -> None:
        left = {
            "status": "OK",
            "objects": [
                {
                    "cluster_id": "seating_hub",
                    "object_id": "sectional_sofa",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 0, "min_y": 0, "max_x": 1000, "max_y": 600},
                },
                {
                    "cluster_id": "media_zone",
                    "object_id": "tv_console",
                    "rotation_ccw": 180,
                    "bbox": {"min_x": 200, "min_y": 900, "max_x": 1200, "max_y": 1200},
                },
            ],
        }
        right = {
            "status": "OK",
            "objects": [
                {
                    "cluster_id": "seating_zone",
                    "object_id": "sectional_sofa",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 0, "min_y": 0, "max_x": 1000, "max_y": 600},
                },
                {
                    "cluster_id": "media_storage_zone",
                    "object_id": "tv_console",
                    "rotation_ccw": 180,
                    "bbox": {"min_x": 200, "min_y": 900, "max_x": 1200, "max_y": 1200},
                },
            ],
        }

        self.assertTrue(layout_variants._is_near_duplicate_layout(left, right))
        self.assertEqual(
            layout_variants._absolute_layout_signature(left),
            layout_variants._absolute_layout_signature(right),
        )

    def test_select_diverse_candidates_skips_near_duplicate_neighbor(self) -> None:
        anchor = _candidate(
            source="phase2_selected",
            reason="selected",
            score=-1000,
            layout=_layout({"living": (0, 0), "media": (900, 0)}),
        )
        near_neighbor = _candidate(
            source="cluster_pose",
            reason="small nudge",
            score=-995,
            layout=_layout({"living": (40, 0), "media": (940, 0)}),
        )
        far_pose = _candidate(
            source="cluster_pose",
            reason="move away from entry",
            score=-1030,
            layout=_layout({"living": (700, 0), "media": (1600, 0)}),
        )
        variant_swap = _candidate(
            source="cluster_variant",
            reason="mirror focal pair",
            score=-1040,
            layout=_layout({"living": (0, 700), "media": (900, 700)}),
        )

        selected = layout_variants._select_diverse_candidates(
            [anchor, near_neighbor, far_pose, variant_swap],
            max_variants=3,
        )

        self.assertEqual(selected[0].source, "phase2_selected")
        selected_sources = {row.source for row in selected}
        self.assertIn("cluster_variant", selected_sources)
        self.assertNotIn(near_neighbor, selected)
        self.assertIn(far_pose, selected)

    def test_macro_region_signature_collapses_small_same_region_moves(self) -> None:
        anchor = {
            "status": "OK",
            "room": {
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 4000},
                    {"x": 0, "y": 4000},
                ]
            },
            "objects": [
                {
                    "cluster_id": "media_zone",
                    "object_id": "tv_console",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 5000, "min_y": 0, "max_x": 5600, "max_y": 400},
                },
                {
                    "cluster_id": "seating_zone",
                    "object_id": "sofa",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 2800, "min_y": 700, "max_x": 4600, "max_y": 1500},
                },
                {
                    "cluster_id": "utility_storage",
                    "object_id": "cabinet",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 700, "min_y": 3100, "max_x": 3000, "max_y": 3500},
                },
            ],
        }
        near = {
            "status": "OK",
            "room": anchor["room"],
            "objects": [
                {
                    "cluster_id": "media_zone",
                    "object_id": "tv_console",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 5000, "min_y": 0, "max_x": 5600, "max_y": 400},
                },
                {
                    "cluster_id": "seating_zone",
                    "object_id": "sofa",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 3000, "min_y": 700, "max_x": 4800, "max_y": 1500},
                },
                {
                    "cluster_id": "utility_storage",
                    "object_id": "cabinet",
                    "rotation_ccw": 0,
                    "bbox": {"min_x": 700, "min_y": 3300, "max_x": 3000, "max_y": 3700},
                },
            ],
        }

        self.assertEqual(
            layout_variants._macro_region_signature(anchor),
            layout_variants._macro_region_signature(near),
        )
        self.assertTrue(layout_variants._is_near_duplicate_layout(anchor, near))

    def test_select_diverse_candidates_prefers_new_macro_region_signature(self) -> None:
        room = {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        }
        anchor = _candidate(
            source="phase2_selected",
            reason="selected",
            score=-1000,
            layout={
                "status": "OK",
                "room": room,
                "objects": [
                    {
                        "cluster_id": "media_zone",
                        "object_id": "tv_console",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 5000,
                            "min_y": 0,
                            "max_x": 5600,
                            "max_y": 400,
                        },
                    },
                    {
                        "cluster_id": "seating_zone",
                        "object_id": "sofa",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 2800,
                            "min_y": 700,
                            "max_x": 4600,
                            "max_y": 1500,
                        },
                    },
                    {
                        "cluster_id": "utility_storage",
                        "object_id": "cabinet",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 700,
                            "min_y": 3100,
                            "max_x": 3000,
                            "max_y": 3500,
                        },
                    },
                ],
            },
        )
        same_region = _candidate(
            source="cluster_pose",
            reason="Move utility_storage toward the room edge",
            score=-900,
            layout={
                "status": "OK",
                "room": room,
                "objects": [
                    {
                        "cluster_id": "media_zone",
                        "object_id": "tv_console",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 5000,
                            "min_y": 0,
                            "max_x": 5600,
                            "max_y": 400,
                        },
                    },
                    {
                        "cluster_id": "seating_zone",
                        "object_id": "sofa",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 2900,
                            "min_y": 700,
                            "max_x": 4700,
                            "max_y": 1500,
                        },
                    },
                    {
                        "cluster_id": "utility_storage",
                        "object_id": "cabinet",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 700,
                            "min_y": 3300,
                            "max_x": 3000,
                            "max_y": 3700,
                        },
                    },
                ],
            },
        )
        new_region = _candidate(
            source="cluster_pose",
            reason="Move seating_zone away from the entry zone",
            score=-1300,
            layout={
                "status": "OK",
                "room": room,
                "objects": [
                    {
                        "cluster_id": "media_zone",
                        "object_id": "tv_console",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 5000,
                            "min_y": 0,
                            "max_x": 5600,
                            "max_y": 400,
                        },
                    },
                    {
                        "cluster_id": "seating_zone",
                        "object_id": "sofa",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 2100,
                            "min_y": 200,
                            "max_x": 3900,
                            "max_y": 1000,
                        },
                    },
                    {
                        "cluster_id": "utility_storage",
                        "object_id": "cabinet",
                        "rotation_ccw": 0,
                        "bbox": {
                            "min_x": 700,
                            "min_y": 3100,
                            "max_x": 3000,
                            "max_y": 3500,
                        },
                    },
                ],
            },
        )

        selected = layout_variants._select_diverse_candidates(
            [anchor, same_region, new_region],
            max_variants=2,
        )

        self.assertEqual(selected[0], anchor)
        self.assertEqual(selected[1], new_region)

    def test_build_variant_style_plan_diversifies_palette_from_styled_result(
        self,
    ) -> None:
        base_styled_result = {
            "room": {
                "surfaces": {
                    "wall_color_hex": "#EDE3D6",
                    "floor_color_hex": "#AF7F52",
                    "ceiling_color_hex": "#FAF6EF",
                },
                "opening_colors": {
                    "doors": [{"id": "door_1", "color_hex": "#8F7259"}],
                    "windows": [{"id": "window_1", "color_hex": "#E6D8C8"}],
                },
            },
            "objects": [
                {
                    "instance_id": "sofa_1",
                    "color_hex": "#C2B5A3",
                    "material": "linen",
                },
                {
                    "instance_id": "console_1",
                    "color_hex": "#8D6B4D",
                    "material": "oak_wood",
                },
            ],
            "notes": ["Base palette."],
        }

        base_plan = layout_variants._build_variant_style_plan(
            base_style_plan={},
            base_styled_result=base_styled_result,
            variant_index=0,
        )
        cooler_plan = layout_variants._build_variant_style_plan(
            base_style_plan={},
            base_styled_result=base_styled_result,
            variant_index=2,
        )

        self.assertEqual(
            base_plan["room_surfaces"]["wall_color_hex"],
            "#EDE3D6",
        )
        self.assertEqual(
            [row["instance_id"] for row in cooler_plan["object_styles"]],
            ["sofa_1", "console_1"],
        )
        self.assertNotEqual(
            cooler_plan["room_surfaces"]["wall_color_hex"],
            base_plan["room_surfaces"]["wall_color_hex"],
        )
        self.assertNotEqual(
            cooler_plan["opening_colors"]["doors"][0]["color_hex"],
            base_plan["opening_colors"]["doors"][0]["color_hex"],
        )
        self.assertNotEqual(
            cooler_plan["opening_colors"]["windows"][0]["color_hex"],
            base_plan["opening_colors"]["windows"][0]["color_hex"],
        )
        self.assertNotEqual(
            cooler_plan["object_styles"][0]["color_hex"],
            base_plan["object_styles"][0]["color_hex"],
        )
        self.assertIn("Variant palette:", cooler_plan["notes"][-1])

    def test_build_final_layout_variants_payload_uses_distinct_stylist_models(
        self,
    ) -> None:
        candidates = [
            {
                "absolute_layout": _layout({"living": (0, 0), "media": (900, 0)}),
                "source": "seed_stable",
                "reason": "Stable seed.",
                "layout_score": 100,
                "hard_valid": True,
                "notes": [],
                "family": "seed_stable",
            },
            {
                "absolute_layout": _layout({"living": (0, 900), "media": (900, 900)}),
                "source": "seed_focal_axis",
                "reason": "Focal axis.",
                "layout_score": 90,
                "hard_valid": True,
                "notes": [],
                "family": "seed_focal_axis",
            },
            {
                "absolute_layout": _layout({"living": (0, 1800), "media": (900, 1800)}),
                "source": "seed_open_center",
                "reason": "Open center.",
                "layout_score": 80,
                "hard_valid": True,
                "notes": [],
                "family": "seed_open_center",
            },
        ]

        class _FakeStylist:
            def __init__(self) -> None:
                self.model_names: list[str | None] = []

            def generate_style_plan(self, **kwargs: object) -> dict[str, object]:
                self.model_names.append(kwargs.get("model_name"))
                return {
                    "notes": [str(kwargs.get("model_name") or "")],
                }

            def apply_style_plan(self, **kwargs: object) -> dict[str, object]:
                style_plan = kwargs.get("style_plan")
                notes = style_plan.get("notes") if isinstance(style_plan, dict) else []
                return {
                    "status": "OK",
                    "room": {
                        "surfaces": {
                            "wall_color_hex": "#EFE7DB",
                            "floor_color_hex": "#AF7F52",
                            "ceiling_color_hex": "#FAF6EF",
                        },
                        "opening_colors": {"doors": [], "windows": []},
                    },
                    "objects": [],
                    "notes": list(notes) if isinstance(notes, list) else [],
                }

        stylist = _FakeStylist()
        payload = layout_variants.build_final_layout_variants_payload(
            candidates=candidates,
            stylist=stylist,
            stylist_user_context_json={},
            stylist_tenant_id=None,
            max_variants=3,
        )

        self.assertEqual(
            stylist.model_names,
            [
                "gemma-4-31b-it",
                "gemini-3.1-flash-lite-preview",
                "gemma-3-27b-it",
            ],
        )
        self.assertEqual(
            payload["variants"][0]["styled_result"]["notes"][0],
            "gemma-4-31b-it",
        )
        self.assertEqual(
            payload["variants"][1]["styled_result"]["notes"][0],
            "gemini-3.1-flash-lite-preview",
        )
        self.assertEqual(
            payload["variants"][2]["styled_result"]["notes"][0],
            "gemma-3-27b-it",
        )

    def test_build_layout_variants_artifact_includes_external_intent_candidates(
        self,
    ) -> None:
        base_layout = _layout({"living": (0, 0), "media": (900, 0)})
        external_layout = _layout({"living": (0, 900), "media": (900, 900)})

        layout_variants.build_phase2_payload = lambda *args, **kwargs: {}
        layout_variants.make_no_improvement_repair = lambda *args, **kwargs: {
            "notes": []
        }
        layout_variants.EvaluatePhase2Proposal = lambda *args, **kwargs: {
            "hard_valid": True,
            "baseline_comparison": {"candidate_score": -100},
        }
        layout_variants.PromotePhase2RepairToSeedPayload = lambda *args, **kwargs: {}
        layout_variants.EnumeratePhase2RepairMoves = lambda *args, **kwargs: {
            "moves": []
        }
        layout_variants.compile_phase2_to_final_output = lambda payload: base_layout

        class _FakeStylist:
            def apply_style_plan(self, **kwargs: object) -> dict[str, object]:
                _ = kwargs
                return {
                    "status": "OK",
                    "room": {
                        "surfaces": {
                            "wall_color_hex": "#EFE7DB",
                            "floor_color_hex": "#AF7F52",
                            "ceiling_color_hex": "#FAF6EF",
                        },
                        "opening_colors": {"doors": [], "windows": []},
                    },
                    "objects": [],
                    "notes": [],
                }

        artifact = layout_variants.build_layout_variants_artifact(
            room_output={},
            merged_output={},
            cluster_outlines={},
            relation_plan={},
            solver_output={},
            phase2_result={
                "proposal": {"notes": []},
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"candidate_score": -100},
                },
            },
            absolute_layout=base_layout,
            stylist_output={
                "status": "OK",
                "room": {
                    "surfaces": {
                        "wall_color_hex": "#EFE7DB",
                        "floor_color_hex": "#AF7F52",
                        "ceiling_color_hex": "#FAF6EF",
                    },
                    "opening_colors": {"doors": [], "windows": []},
                },
                "objects": [],
                "notes": [],
            },
            stylist=_FakeStylist(),
            stylist_user_context_json={},
            stylist_tenant_id=None,
            style_plan={},
            external_candidates=[
                {
                    "absolute_layout": external_layout,
                    "source": "intent_branch:intent_2",
                    "reason": "Open-center branch.",
                    "layout_score": -90,
                    "hard_valid": True,
                    "notes": ["Intent branch."],
                    "family": "intent_branch:intent_2",
                }
            ],
            max_variants=2,
        )

        sources = {row["source"] for row in artifact["variants"]}
        self.assertIn("intent_branch:intent_2", sources)

    def test_build_layout_variants_artifact_reports_incremental_progress(self) -> None:
        base_layout = _layout({"living": (0, 0), "media": (900, 0)})
        external_layout = _layout({"living": (0, 900), "media": (900, 900)})

        layout_variants.build_phase2_payload = lambda *args, **kwargs: {}
        layout_variants.make_no_improvement_repair = lambda *args, **kwargs: {
            "notes": []
        }
        layout_variants.EvaluatePhase2Proposal = lambda *args, **kwargs: {
            "hard_valid": True,
            "baseline_comparison": {"candidate_score": -100},
        }
        layout_variants.PromotePhase2RepairToSeedPayload = lambda *args, **kwargs: {}
        layout_variants.EnumeratePhase2RepairMoves = lambda *args, **kwargs: {
            "moves": []
        }
        layout_variants.compile_phase2_to_final_output = lambda payload: base_layout

        class _FakeStylist:
            def apply_style_plan(self, **kwargs: object) -> dict[str, object]:
                _ = kwargs
                return {
                    "status": "OK",
                    "room": {
                        "surfaces": {
                            "wall_color_hex": "#EFE7DB",
                            "floor_color_hex": "#AF7F52",
                            "ceiling_color_hex": "#FAF6EF",
                        },
                        "opening_colors": {"doors": [], "windows": []},
                    },
                    "objects": [],
                    "notes": [],
                }

        progress_payloads: list[dict[str, object]] = []
        artifact = layout_variants.build_layout_variants_artifact(
            room_output={},
            merged_output={},
            cluster_outlines={},
            relation_plan={},
            solver_output={},
            phase2_result={
                "proposal": {"notes": []},
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"candidate_score": -100},
                },
            },
            absolute_layout=base_layout,
            stylist_output={
                "status": "OK",
                "room": {
                    "surfaces": {
                        "wall_color_hex": "#EFE7DB",
                        "floor_color_hex": "#AF7F52",
                        "ceiling_color_hex": "#FAF6EF",
                    },
                    "opening_colors": {"doors": [], "windows": []},
                },
                "objects": [],
                "notes": [],
            },
            stylist=_FakeStylist(),
            stylist_user_context_json={},
            stylist_tenant_id=None,
            style_plan={},
            external_candidates=[
                {
                    "absolute_layout": external_layout,
                    "source": "intent_branch:intent_2",
                    "reason": "Open-center branch.",
                    "layout_score": -90,
                    "hard_valid": True,
                    "notes": ["Intent branch."],
                    "family": "intent_branch:intent_2",
                }
            ],
            max_variants=2,
            on_progress=lambda payload: progress_payloads.append(payload),
        )

        self.assertEqual(artifact["status"], "OK")
        self.assertEqual(len(progress_payloads), 2)
        self.assertEqual(progress_payloads[0]["status"], "BUILDING")
        self.assertEqual(len(progress_payloads[0]["variants"]), 1)
        self.assertEqual(len(progress_payloads[1]["variants"]), 2)

    def test_build_layout_variants_artifact_skips_internal_search_when_external_pool_is_ready(
        self,
    ) -> None:
        base_layout = _layout({"living": (0, 0), "media": (900, 0)})
        external_layouts = [
            _layout({"living": (0, 900), "media": (900, 900)}),
            _layout({"living": (900, 0), "media": (1800, 0)}),
            _layout({"living": (0, 1800), "media": (900, 1800)}),
            _layout({"living": (1800, 0), "media": (2700, 0)}),
        ]

        layout_variants.build_phase2_payload = lambda *args, **kwargs: {}
        layout_variants.make_no_improvement_repair = lambda *args, **kwargs: {
            "notes": []
        }
        layout_variants.EvaluatePhase2Proposal = lambda *args, **kwargs: {
            "hard_valid": True,
            "baseline_comparison": {"candidate_score": -100},
        }
        layout_variants.PromotePhase2RepairToSeedPayload = lambda *args, **kwargs: {}
        enumerate_calls = {"count": 0}

        def _counting_enum(*args: object, **kwargs: object) -> dict[str, object]:
            _ = args
            _ = kwargs
            enumerate_calls["count"] += 1
            return {"moves": []}

        layout_variants.EnumeratePhase2RepairMoves = _counting_enum
        layout_variants.compile_phase2_to_final_output = lambda payload: base_layout

        class _FakeStylist:
            def apply_style_plan(self, **kwargs: object) -> dict[str, object]:
                _ = kwargs
                return {
                    "status": "OK",
                    "room": {
                        "surfaces": {
                            "wall_color_hex": "#EFE7DB",
                            "floor_color_hex": "#AF7F52",
                            "ceiling_color_hex": "#FAF6EF",
                        },
                        "opening_colors": {"doors": [], "windows": []},
                    },
                    "objects": [],
                    "notes": [],
                }

        artifact = layout_variants.build_layout_variants_artifact(
            room_output={},
            merged_output={},
            cluster_outlines={},
            relation_plan={},
            solver_output={},
            phase2_result={
                "proposal": {"notes": []},
                "tool_evaluation": {
                    "hard_valid": True,
                    "baseline_comparison": {"candidate_score": -100},
                },
            },
            absolute_layout=base_layout,
            stylist_output={
                "status": "OK",
                "room": {
                    "surfaces": {
                        "wall_color_hex": "#EFE7DB",
                        "floor_color_hex": "#AF7F52",
                        "ceiling_color_hex": "#FAF6EF",
                    },
                    "opening_colors": {"doors": [], "windows": []},
                },
                "objects": [],
                "notes": [],
            },
            stylist=_FakeStylist(),
            stylist_user_context_json={},
            stylist_tenant_id=None,
            style_plan={},
            external_candidates=[
                {
                    "absolute_layout": external_layout,
                    "source": f"intent_branch:intent_{index + 2}",
                    "reason": "Intent-derived preview.",
                    "layout_score": -90 + index,
                    "hard_valid": True,
                    "notes": ["Intent branch."],
                    "family": f"intent_branch:intent_{index + 2}",
                }
                for index, external_layout in enumerate(external_layouts)
            ],
            max_variants=5,
        )

        self.assertEqual(enumerate_calls["count"], 0)
        self.assertEqual(len(artifact["variants"]), 5)


if __name__ == "__main__":
    unittest.main()
