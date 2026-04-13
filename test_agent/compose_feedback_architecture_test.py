from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

compose_feedback = importlib.import_module("pipeline.compose_feedback")


def _room_output() -> dict[str, object]:
    return {
        "room": {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 3000},
                {"x": 3500, "y": 3000},
                {"x": 3500, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        }
    }


def _tier_output() -> dict[str, object]:
    return {
        "status": "OK",
        "notes": ["Initial deterministic tier-count output."],
        "decisions": [
            {
                "object_type": "sectional_sofa",
                "cluster_id": "seating_hub",
                "quantity": 1,
                "size_tier": "S",
                "priority": "anchor",
                "rationale": "anchor",
                "rep_dims_m": {"A": 2.3},
            },
            {
                "object_type": "sofa",
                "cluster_id": "seating_hub",
                "quantity": 1,
                "size_tier": "S",
                "priority": "primary",
                "rationale": "primary",
                "rep_dims_m": {"A": 1.4},
            },
            {
                "object_type": "armchair",
                "cluster_id": "seating_hub",
                "quantity": 1,
                "size_tier": "S",
                "priority": "primary",
                "rationale": "primary",
                "rep_dims_m": {"A": 0.6},
            },
            {
                "object_type": "ottoman",
                "cluster_id": "seating_hub",
                "quantity": 1,
                "size_tier": "S",
                "priority": "secondary",
                "rationale": "secondary",
                "rep_dims_m": {"A": 0.3},
            },
        ],
    }


def _merged_output() -> dict[str, object]:
    tier = _tier_output()
    return {
        "clusters": [
            {
                "cluster_id": "seating_hub",
                "anchors": ["sectional_sofa"],
                "decisions": tier["decisions"],
            }
        ]
    }


class ComposeFeedbackArchitectureTest(unittest.TestCase):
    def test_evaluate_composed_cluster_feasibility_flags_envelope_overflow(
        self,
    ) -> None:
        feasibility = compose_feedback.evaluate_composed_cluster_feasibility(
            room_output=_room_output(),
            merged_output=_merged_output(),
            cluster_results={"seating_hub": {"status": "OK"}},
            cluster_outlines={
                "seating_hub": {
                    "status": "OK",
                    "cluster_footprint": {
                        "local_bbox": {
                            "min_x": 0,
                            "min_y": 0,
                            "max_x": 7200,
                            "max_y": 1800,
                        }
                    },
                }
            },
        )

        self.assertFalse(feasibility["feasible"])
        self.assertEqual(
            feasibility["offenders"][0]["reason"],
            "CLUSTER_ENVELOPE_EXCEEDS_ROOM",
        )

    def test_apply_compose_backoff_drops_secondary_before_primary(self) -> None:
        updated, notes, changed = compose_feedback.apply_compose_backoff(
            tier_output=_tier_output(),
            merged_output=_merged_output(),
            feasibility={
                "offenders": [
                    {
                        "cluster_id": "seating_hub",
                        "reason": "CLUSTER_ENVELOPE_EXCEEDS_ROOM",
                    }
                ]
            },
        )

        self.assertTrue(changed)
        decisions = updated["decisions"]
        by_id = {row["object_type"]: row for row in decisions}
        self.assertEqual(by_id["ottoman"]["quantity"], 0)
        self.assertEqual(by_id["sofa"]["quantity"], 1)
        self.assertTrue(notes)

    def test_evaluate_composed_cluster_feasibility_flags_polygon_area_overflow(
        self,
    ) -> None:
        feasibility = compose_feedback.evaluate_composed_cluster_feasibility(
            room_output=_room_output(),
            merged_output=_merged_output(),
            cluster_results={"seating_hub": {"status": "OK"}},
            cluster_outlines={
                "seating_hub": {
                    "status": "OK",
                    "cluster_footprint": {
                        "rects": [
                            {
                                "id": "sectional_sofa",
                                "x": 0,
                                "y": 0,
                                "w": 5800,
                                "h": 3800,
                            }
                        ],
                        "local_bbox": {
                            "min_x": 0,
                            "min_y": 0,
                            "max_x": 5800,
                            "max_y": 3800,
                        },
                    },
                }
            },
        )

        self.assertFalse(feasibility["feasible"])
        self.assertEqual(
            feasibility["offenders"][0]["reason"],
            "CLUSTER_ENVELOPE_EXCEEDS_ROOM",
        )
        self.assertEqual(
            feasibility["offenders"][0]["fit_failure_mode"],
            "polygon_area_upper_bound",
        )

    def test_evaluate_solver_cluster_feasibility_flags_skipped_cluster(self) -> None:
        feasibility = compose_feedback.evaluate_solver_cluster_feasibility(
            merged_output=_merged_output(),
            solver_output={
                "status": "UNSAT",
                "skipped_clusters": ["seating_hub"],
                "solver_debug": {
                    "candidate_counts": {
                        "seating_hub": 0,
                    }
                },
            },
        )

        self.assertFalse(feasibility["feasible"])
        self.assertEqual(
            feasibility["offenders"][0]["reason"],
            "SOLVER_HAS_NO_HARD_VALID_CANDIDATES",
        )
        self.assertEqual(feasibility["offenders"][0]["cluster_id"], "seating_hub")

    def test_evaluate_solver_cluster_feasibility_flags_unsat_without_skips(
        self,
    ) -> None:
        feasibility = compose_feedback.evaluate_solver_cluster_feasibility(
            merged_output=_merged_output(),
            solver_output={
                "status": "UNSAT",
                "skipped_clusters": [],
                "solver_debug": {
                    "candidate_counts": {
                        "seating_hub": 24,
                    },
                    "best_verify": {
                        "repair_guidance": {
                            "prioritized_clusters": [
                                {"cluster_id": "seating_hub"},
                            ]
                        }
                    },
                },
            },
        )

        self.assertFalse(feasibility["feasible"])
        self.assertEqual(
            feasibility["offenders"][0]["reason"],
            "SOLVER_EXACT_ASSIGNMENT_UNSAT",
        )
        self.assertEqual(feasibility["offenders"][0]["cluster_id"], "seating_hub")

    def test_evaluate_solver_cluster_feasibility_accepts_complete_hard_valid_seed(
        self,
    ) -> None:
        feasibility = compose_feedback.evaluate_solver_cluster_feasibility(
            merged_output=_merged_output(),
            solver_output={
                "status": "UNSAT",
                "skipped_clusters": [],
                "solver_debug": {
                    "candidate_counts": {
                        "seating_hub": 24,
                    },
                    "best_verify": {
                        "hard_valid": True,
                        "complete": True,
                        "acceptable_valid": False,
                        "quality_gate": {
                            "reasons": [
                                "critical_orientation_penalty_too_high",
                            ]
                        },
                    },
                },
            },
        )

        self.assertTrue(feasibility["feasible"])
        self.assertEqual(feasibility["offenders"], [])
        self.assertEqual(feasibility["solver_status"], "UNSAT")

    def test_evaluate_solver_cluster_feasibility_keeps_ranked_unsat_fallbacks(
        self,
    ) -> None:
        feasibility = compose_feedback.evaluate_solver_cluster_feasibility(
            merged_output={
                "clusters": [
                    {
                        "cluster_id": "media_zone",
                        "anchors": ["tv_console"],
                        "decisions": [
                            {
                                "object_type": "tv_console",
                                "cluster_id": "media_zone",
                                "quantity": 1,
                                "size_tier": "S",
                                "priority": "anchor",
                                "rep_dims_m": {"A": 0.4},
                            }
                        ],
                    },
                    {
                        "cluster_id": "seating_hub",
                        "anchors": ["sectional_sofa"],
                        "decisions": [
                            {
                                "object_type": "sectional_sofa",
                                "cluster_id": "seating_hub",
                                "quantity": 1,
                                "size_tier": "L",
                                "priority": "anchor",
                                "rep_dims_m": {"A": 2.3},
                            }
                        ],
                    },
                ]
            },
            solver_output={
                "status": "UNSAT",
                "skipped_clusters": [],
                "solver_debug": {
                    "candidate_counts": {
                        "media_zone": 24,
                        "seating_hub": 24,
                    },
                    "best_verify": {
                        "repair_guidance": {
                            "prioritized_clusters": [
                                {"cluster_id": "media_zone"},
                                {"cluster_id": "seating_hub"},
                            ]
                        }
                    },
                },
            },
        )

        self.assertEqual(
            [item["cluster_id"] for item in feasibility["offenders"]],
            ["media_zone", "seating_hub"],
        )

    def test_apply_compose_backoff_shrinks_single_anchor_cluster(self) -> None:
        tier_output = {
            "status": "OK",
            "decisions": [
                {
                    "object_type": "sectional_sofa",
                    "cluster_id": "seating_hub",
                    "quantity": 1,
                    "size_tier": "L",
                    "priority": "anchor",
                    "rationale": "anchor",
                    "rep_dims_m": {"A": 2.3},
                }
            ],
        }
        merged_output = {
            "clusters": [
                {
                    "cluster_id": "seating_hub",
                    "anchors": ["sectional_sofa"],
                    "decisions": tier_output["decisions"],
                }
            ]
        }

        updated, notes, changed = compose_feedback.apply_compose_backoff(
            tier_output=tier_output,
            merged_output=merged_output,
            feasibility={
                "offenders": [
                    {
                        "cluster_id": "seating_hub",
                        "reason": "SOLVER_EXACT_ASSIGNMENT_UNSAT",
                    }
                ]
            },
        )

        self.assertTrue(changed)
        self.assertEqual(updated["decisions"][0]["quantity"], 1)
        self.assertEqual(updated["decisions"][0]["size_tier"], "M")
        self.assertTrue(notes)

    def test_apply_compose_backoff_uses_next_offender_when_first_is_locked(
        self,
    ) -> None:
        tier_output = {
            "status": "OK",
            "decisions": [
                {
                    "object_type": "tv_console",
                    "cluster_id": "media_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "anchor",
                    "rationale": "anchor",
                    "rep_dims_m": {"A": 0.4},
                },
                {
                    "object_type": "sectional_sofa",
                    "cluster_id": "seating_hub",
                    "quantity": 1,
                    "size_tier": "L",
                    "priority": "anchor",
                    "rationale": "anchor",
                    "rep_dims_m": {"A": 2.3},
                },
            ],
        }
        merged_output = {
            "clusters": [
                {
                    "cluster_id": "media_zone",
                    "anchors": ["tv_console"],
                    "decisions": [tier_output["decisions"][0]],
                },
                {
                    "cluster_id": "seating_hub",
                    "anchors": ["sectional_sofa"],
                    "decisions": [tier_output["decisions"][1]],
                },
            ]
        }

        updated, notes, changed = compose_feedback.apply_compose_backoff(
            tier_output=tier_output,
            merged_output=merged_output,
            feasibility={
                "offenders": [
                    {
                        "cluster_id": "media_zone",
                        "reason": "SOLVER_EXACT_ASSIGNMENT_UNSAT",
                    },
                    {
                        "cluster_id": "seating_hub",
                        "reason": "SOLVER_EXACT_ASSIGNMENT_UNSAT",
                    },
                ]
            },
        )

        self.assertTrue(changed)
        decisions = {str(item["cluster_id"]): item for item in updated["decisions"]}
        self.assertEqual(decisions["media_zone"]["size_tier"], "S")
        self.assertEqual(decisions["seating_hub"]["size_tier"], "M")
        self.assertTrue(notes)

    def test_apply_variant_diversification_changes_seed(self) -> None:
        updated, notes, changed = compose_feedback.apply_variant_diversification(
            tier_output=_tier_output(),
            merged_output=_merged_output(),
            variant_index=2,
        )

        self.assertTrue(changed)
        decisions = updated["decisions"]
        by_id = {row["object_type"]: row for row in decisions}
        self.assertIn(by_id["ottoman"]["quantity"], {0, 1})
        self.assertTrue(
            by_id["ottoman"]["quantity"] == 0
            or by_id["ottoman"]["size_tier"] != "S"
            or by_id["sofa"]["size_tier"] != "S"
        )
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
