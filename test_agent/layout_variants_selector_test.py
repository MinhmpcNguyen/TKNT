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
) -> dict[str, object]:
    return {
        "absolute_layout": layout,
        "styled_result": {"objects": []},
        "source": source,
        "reason": "Object-level layout candidate.",
        "layout_score": score,
        "hard_valid": True,
        "complete": True,
        "gallery_eligible": True,
        "coverage_ratio": 1.0,
        "notes": [],
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


if __name__ == "__main__":
    unittest.main()
