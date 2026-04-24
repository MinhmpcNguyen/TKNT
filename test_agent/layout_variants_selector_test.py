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


def _candidate(source: str, layout: dict[str, object]) -> dict[str, object]:
    return {
        "absolute_layout": layout,
        "styled_result": {"objects": []},
        "source": source,
        "reason": "Object-level layout candidate.",
        "layout_score": 100,
        "hard_valid": True,
        "complete": True,
        "gallery_eligible": True,
        "coverage_ratio": 1.0,
        "notes": [],
    }


class LayoutVariantsSelectorTest(unittest.TestCase):
    def test_selector_does_not_backfill_same_macro_layouts(self) -> None:
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

        self.assertEqual(len(selected), 1)
        self.assertNotIn("exact_duplicate", {row["source"] for row in selected})
        self.assertEqual(
            len({row["state_signature"] for row in selected}),
            len(selected),
        )
        self.assertEqual(
            len({row["macro_layout_signature"] for row in selected}),
            1,
        )
        self.assertEqual(selected[0]["gallery_selection_mode"], "distinct")


if __name__ == "__main__":
    unittest.main()
