from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import layout_variants  # noqa: E402


_ROOM = {
    "polygon_ccw": [
        {"x": 0, "y": 0},
        {"x": 6000, "y": 0},
        {"x": 6000, "y": 4000},
        {"x": 0, "y": 4000},
    ]
}


def _layout(cluster_positions: dict[str, tuple[int, int]]) -> dict[str, object]:
    object_by_cluster = {
        "living": "sofa",
        "media": "tv_console",
        "work": "desk",
        "storage": "cabinet",
    }
    objects: list[dict[str, object]] = []
    for cluster_id, (x_pos, y_pos) in cluster_positions.items():
        object_id = object_by_cluster.get(cluster_id, f"{cluster_id}_object")
        objects.append(
            {
                "cluster_id": cluster_id,
                "object_id": object_id,
                "instance_id": f"{cluster_id}_{object_id}",
                "rotation_ccw": 0,
                "bbox": {
                    "min_x": x_pos,
                    "min_y": y_pos,
                    "max_x": x_pos + 300,
                    "max_y": y_pos + 300,
                },
            }
        )
    return {
        "status": "OK",
        "room": _ROOM,
        "objects": objects,
        "complete": True,
        "gallery_eligible": True,
        "coverage_ratio": 1.0,
    }


def _candidate(
    *,
    source: str,
    layout: dict[str, object],
    score: int = 100,
    cluster_output: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "absolute_layout": layout,
        "source": source,
        "reason": f"{source} candidate.",
        "layout_score": score,
        "hard_valid": True,
        "complete": True,
        "gallery_eligible": True,
        "coverage_ratio": 1.0,
        "cluster_output": cluster_output or {},
    }


class LayoutVariantsArchitectureTest(unittest.TestCase):
    def test_select_distinct_gallery_candidates_skips_duplicate_state(self) -> None:
        base = _layout({"living": (0, 0), "media": (900, 0)})
        duplicate = _layout({"living": (0, 0), "media": (900, 0)})
        distinct = _layout({"living": (0, 3000), "media": (900, 3000)})

        selected = layout_variants.select_distinct_final_gallery_candidates(
            candidates=[
                _candidate(source="base", layout=base, score=100),
                _candidate(source="duplicate", layout=duplicate, score=95),
                _candidate(source="distinct", layout=distinct, score=90),
            ],
            max_variants=3,
        )

        self.assertEqual([row["source"] for row in selected], ["base", "distinct"])
        self.assertEqual(selected[0]["gallery_selection_mode"], "distinct")

    def test_material_macro_layout_signature_collapses_small_region_moves(self) -> None:
        anchor = _layout({"media": (5000, 0), "living": (2800, 700)})
        near = _layout({"media": (5000, 0), "living": (3000, 700)})
        far = _layout({"media": (0, 3000), "living": (3000, 3000)})

        self.assertEqual(
            layout_variants.material_macro_layout_signature(anchor),
            layout_variants.material_macro_layout_signature(near),
        )
        self.assertNotEqual(
            layout_variants.material_macro_layout_signature(anchor),
            layout_variants.material_macro_layout_signature(far),
        )

    def test_missing_hard_requested_solver_object_blocks_gallery(self) -> None:
        cluster_output = {
            "decisions": [
                {
                    "object_type": "desk",
                    "target_count": 1,
                    "intent": "must_keep",
                    "solver_participation": "hard_solver_object",
                }
            ]
        }

        payload = layout_variants.build_final_layout_variants_payload(
            candidates=[
                _candidate(
                    source="missing_required_desk",
                    layout=_layout({"living": (0, 0)}),
                    cluster_output=cluster_output,
                )
            ],
            max_variants=1,
        )

        self.assertEqual(payload["variants"], [])
        self.assertEqual(payload["selection_summary"]["eligible_candidate_count"], 0)
        self.assertIn(
            "Final gallery returned no options",
            " ".join(payload["selection_summary"]["notes"]),
        )

    def test_build_final_layout_variants_payload_uses_current_candidate_payload(
        self,
    ) -> None:
        payload = layout_variants.build_final_layout_variants_payload(
            candidates=[
                _candidate(
                    source="solver:concept_1",
                    layout=_layout({"living": (0, 0), "media": (900, 0)}),
                    score=100,
                ),
                _candidate(
                    source="intent_branch:intent_2",
                    layout=_layout({"living": (0, 3000), "media": (900, 3000)}),
                    score=90,
                ),
            ],
            max_variants=2,
        )

        self.assertEqual(payload["status"], "OK")
        self.assertEqual(
            [row["source"] for row in payload["variants"]],
            ["solver:concept_1", "intent_branch:intent_2"],
        )


if __name__ == "__main__":
    unittest.main()
