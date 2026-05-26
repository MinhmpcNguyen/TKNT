from __future__ import annotations

import unittest

from pipeline import layout_variants


def _layout(x_pos: int) -> dict[str, object]:
    return {
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
                "cluster_id": "main_seating",
                "object_id": "sofa",
                "rotation_ccw": 0,
                "bbox": {
                    "min_x": x_pos,
                    "min_y": 100,
                    "max_x": x_pos + 300,
                    "max_y": 700,
                },
            }
        ],
    }


def _candidate(
    source: str,
    layout: dict[str, object],
    *,
    score: int = 100,
    hard_valid: bool = True,
    gallery_eligible: bool = True,
) -> dict[str, object]:
    return {
        "absolute_layout": layout,
        "styled_result": {"objects": []},
        "source": source,
        "reason": "Object-level layout candidate.",
        "layout_score": score,
        "hard_valid": hard_valid,
        "complete": True,
        "gallery_eligible": gallery_eligible,
        "coverage_ratio": 1.0,
        "notes": [],
    }


def _layout_with_object_types(object_types: list[str]) -> dict[str, object]:
    layout = _layout(0)
    layout["objects"] = [
        {
            "cluster_id": "main_seating",
            "object_id": object_type,
            "object_type": object_type,
            "rotation_ccw": 0,
            "bbox": {
                "min_x": 200 + index * 650,
                "min_y": 100,
                "max_x": 600 + index * 650,
                "max_y": 700,
            },
        }
        for index, object_type in enumerate(object_types)
    ]
    return layout


def _bedroom_layout_with_bookshelf(x_pos: int) -> dict[str, object]:
    layout = _layout(0)
    layout["objects"] = [
        {
            "cluster_id": "sleep_core",
            "object_id": "bed",
            "object_type": "bed",
            "role": "dominant_anchor",
            "rotation_ccw": 0,
            "bbox": {"min_x": 1000, "min_y": 300, "max_x": 2600, "max_y": 2300},
        },
        {
            "cluster_id": "storage_core",
            "object_id": "wardrobe",
            "object_type": "wardrobe",
            "rotation_ccw": 180,
            "bbox": {"min_x": 3600, "min_y": 3000, "max_x": 4800, "max_y": 3600},
        },
        {
            "cluster_id": "storage_core",
            "object_id": "bookshelf",
            "object_type": "bookshelf",
            "rotation_ccw": 0,
            "bbox": {
                "min_x": x_pos,
                "min_y": 3000,
                "max_x": x_pos + 700,
                "max_y": 3400,
            },
        },
    ]
    return layout


def _tier_output_for_requested_living_room() -> dict[str, object]:
    return {
        "decisions": [
            {
                "object_type": "sofa",
                "quantity": 1,
                "preserve_level": "highest",
                "role": "dominant_anchor",
            },
            {
                "object_type": "coffee_table",
                "quantity": 1,
                "preserve_level": "highest",
                "role": "support",
            },
            {
                "object_type": "armchair",
                "quantity": 1,
                "preserve_level": "high",
                "role": "secondary_support",
            },
        ]
    }


class LayoutVariantsSelectorTest(unittest.TestCase):
    def test_selector_prefers_distinct_visual_family_before_backfill(self) -> None:
        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[
                _candidate("option_1", _layout(0), score=100),
                _candidate("tiny_nudge", _layout(50), score=110),
                _candidate("option_2", _layout(3600), score=90),
            ],
            max_variants=2,
        )

        self.assertEqual(
            [row["source"] for row in selected], ["tiny_nudge", "option_2"]
        )
        self.assertEqual(
            [row["gallery_selection_mode"] for row in selected],
            ["distinct", "distinct"],
        )

    def test_selector_caps_backfill_to_meaningful_visual_families(self) -> None:
        candidates = [
            _candidate("option_1", _layout(0)),
            _candidate("exact_duplicate", _layout(0)),
            _candidate("tiny_nudge", _layout(50)),
            _candidate("option_2", _layout(120)),
            _candidate("option_3", _layout(250)),
            _candidate("option_4", _layout(380)),
            _candidate("option_5", _layout(510)),
            _candidate("option_6", _layout(650)),
        ]

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=candidates,
            max_variants=5,
        )

        self.assertEqual(len(selected), 2)
        self.assertNotIn("exact_duplicate", {row["source"] for row in selected})
        self.assertEqual(
            len({row["state_signature"] for row in selected}),
            len(selected),
        )
        self.assertEqual(
            [row["gallery_selection_mode"] for row in selected].count("distinct"),
            2,
        )
        self.assertEqual(selected[1]["source"], "option_5")

    def test_selector_does_not_backfill_unseen_concept_when_layout_is_too_similar(
        self,
    ) -> None:
        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[
                _candidate("concept:concept_01", _layout(0), score=100),
                _candidate("concept:concept_01", _layout(260), score=99),
                _candidate("concept:concept_02", _layout(50), score=10),
            ],
            max_variants=2,
        )

        self.assertEqual(
            [row["source"] for row in selected],
            ["concept:concept_01"],
        )
        self.assertEqual(
            [row["gallery_selection_mode"] for row in selected],
            ["distinct"],
        )

    def test_selector_rejects_non_gallery_eligible_candidates(self) -> None:
        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[
                _candidate(
                    "bad",
                    _layout(0),
                    hard_valid=False,
                    gallery_eligible=False,
                    score=999,
                ),
                _candidate("good", _layout(3600), score=100),
            ],
            max_variants=2,
        )

        self.assertEqual([row["source"] for row in selected], ["good"])

    def test_selector_rejects_missing_hard_requested_solver_object(self) -> None:
        missing_tv = _candidate(
            "missing_tv",
            _layout_with_object_types(["sofa"]),
            score=999,
        )
        missing_tv["cluster_output"] = {
            "request_contract": {
                "objects": [
                    {
                        "object_type": "tv_console",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "solver_participation": "hard_solver_object",
                        "blocks_floor_area": True,
                    }
                ]
            }
        }
        covers_tv = _candidate(
            "covers_tv",
            _layout_with_object_types(["sofa", "tv_console"]),
            score=10,
        )
        covers_tv["cluster_output"] = missing_tv["cluster_output"]

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[missing_tv, covers_tv],
            max_variants=1,
        )

        self.assertEqual([row["source"] for row in selected], ["covers_tv"])

    def test_selector_allows_missing_post_solver_requested_accessory(self) -> None:
        missing_mirror = _candidate(
            "missing_mirror",
            _layout_with_object_types(["sofa"]),
            score=100,
        )
        missing_mirror["cluster_output"] = {
            "request_contract": {
                "objects": [
                    {
                        "object_type": "mirror",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "solver_participation": "post_solver_accessory",
                        "blocks_floor_area": False,
                    }
                ]
            }
        }

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[missing_mirror],
            max_variants=1,
        )

        self.assertEqual([row["source"] for row in selected], ["missing_mirror"])

    def test_selector_uses_nested_contract_and_honors_fulfilled_by(self) -> None:
        side_table_fulfilled_by_nightstand = _candidate(
            "nightstand_covers_side_table",
            _layout_with_object_types(["bed", "nightstand"]),
            score=100,
        )
        side_table_fulfilled_by_nightstand["cluster_output"] = {
            "semantic_layout_program": {
                "request_contract": {
                    "objects": [
                        {
                            "object_type": "side_table",
                            "intent": "must_keep",
                            "min_keep": 1,
                            "target_count": 1,
                            "solver_participation": "hard_solver_object",
                            "blocks_floor_area": True,
                            "fulfilled_by": "nightstand",
                        }
                    ]
                }
            }
        }

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[side_table_fulfilled_by_nightstand],
            max_variants=1,
        )

        self.assertEqual(
            [row["source"] for row in selected],
            ["nightstand_covers_side_table"],
        )

    def test_support_only_bookshelf_move_does_not_create_visual_family(self) -> None:
        left = _candidate("left_bookshelf", _bedroom_layout_with_bookshelf(100))
        right = _candidate("right_bookshelf", _bedroom_layout_with_bookshelf(4600))

        summary = layout_variants.build_final_gallery_selection_summary(
            candidates=[left, right],
            selected=[left, right],
            requested_count=2,
        )

        self.assertEqual(summary["visual_family_distinct_returned_count"], 1)

    def test_selection_summary_reports_macro_diversity_degraded(self) -> None:
        selected = [
            _candidate(f"option_{index}", _layout(index * 15)) for index in range(3)
        ]

        summary = layout_variants.build_final_gallery_selection_summary(
            candidates=selected,
            selected=selected,
            requested_count=5,
        )

        self.assertTrue(summary["diversity_degraded"])
        self.assertEqual(summary["material_macro_target_count"], 3)

    def test_selection_summary_reports_gallery_gate_failures(self) -> None:
        bad = _candidate(
            "bad",
            _layout(0),
            hard_valid=False,
            gallery_eligible=False,
        )
        bad["verify_summary"] = {
            "quality_gate_reasons": ["blocking_protected_region_overlap"]
        }

        summary = layout_variants.build_final_gallery_selection_summary(
            candidates=[bad],
            selected=[],
            requested_count=1,
        )

        self.assertEqual(summary["eligible_candidate_count"], 0)
        self.assertEqual(summary["hard_valid_candidate_count"], 0)
        self.assertEqual(summary["gallery_eligible_candidate_count"], 0)
        self.assertEqual(
            summary["top_quality_gate_reasons"],
            {"blocking_protected_region_overlap": 1},
        )
        self.assertIn("failed hard-valid", summary["notes"][0])

    def test_selector_prefers_requested_object_coverage_before_layout_score(
        self,
    ) -> None:
        missing_armchair = _candidate(
            "high_score_missing_armchair",
            _layout_with_object_types(["sofa", "coffee_table"]),
            score=999,
        )
        missing_armchair["tier_output"] = _tier_output_for_requested_living_room()
        covers_armchair = _candidate(
            "lower_score_covers_armchair",
            _layout_with_object_types(["sofa", "coffee_table", "armchair"]),
            score=10,
        )
        covers_armchair["tier_output"] = _tier_output_for_requested_living_room()

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[missing_armchair, covers_armchair],
            max_variants=1,
        )

        self.assertEqual(
            [row["source"] for row in selected], ["lower_score_covers_armchair"]
        )
        self.assertGreater(selected[0]["request_coverage_score"], 1000)


if __name__ == "__main__":
    unittest.main()
