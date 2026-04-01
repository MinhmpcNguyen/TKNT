from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_ortools = types.ModuleType("ortools")
fake_sat = types.ModuleType("ortools.sat")
fake_python = types.ModuleType("ortools.sat.python")
fake_cp_model = types.ModuleType("ortools.sat.python.cp_model")
fake_cp_model.IntVar = object
fake_cp_model.CpModel = object
fake_cp_model.CpSolver = object
fake_cp_model.OPTIMAL = 4
fake_cp_model.FEASIBLE = 2
sys.modules.setdefault("ortools", fake_ortools)
sys.modules.setdefault("ortools.sat", fake_sat)
sys.modules.setdefault("ortools.sat.python", fake_python)
sys.modules.setdefault("ortools.sat.python.cp_model", fake_cp_model)

_solver = importlib.import_module("agent.solver.solver")
Candidate = _solver.Candidate


class SolverArchitectureTest(unittest.TestCase):
    def test_skip_clusters_without_candidates_filters_related_payloads(self) -> None:
        sleep_candidate = Candidate(
            cluster_id="sleep",
            variant_id="sleep_base",
            variant_family="base",
            variant_priority=1.0,
            x=0,
            y=0,
            rot=0,
            anchor_kind="seed",
            anchor_priority=1.0,
            stage="seed",
            hard_valid=True,
            acceptable_valid=True,
            rough_score=100,
            orientation_penalty_mm=0,
            critical_orientation_penalty_mm=0,
            focal_orientation_penalty_mm=0,
            quality_gate_reasons=(),
            hard_error_codes=(),
            state_signature="sleep@0,0,0",
        )
        (
            filtered_outlines,
            filtered_relation_plan,
            filtered_constraints,
            filtered_candidates,
            filtered_variants,
            filtered_rotations,
            skipped_ids,
        ) = _solver._skip_clusters_without_candidates(
            clusters_outlines={
                "sleep": {"status": "OK", "cluster_id": "sleep"},
                "work": {"status": "OK", "cluster_id": "work"},
            },
            relation_plan={
                "cluster_relations": [{"a": "sleep", "b": "work"}],
                "cluster_affinities": [{"cluster_id": "work"}],
            },
            cluster_constraints={
                "clusters": [
                    {"cluster_id": "sleep"},
                    {"cluster_id": "work"},
                ]
            },
            candidates_by_cluster={
                "sleep": [sleep_candidate],
                "work": [],
            },
            variant_payload_lookup={
                "sleep": {"sleep_base": {"cluster_id": "sleep"}},
                "work": {"work_base": {"cluster_id": "work"}},
            },
            cluster_allowed_rotations={
                "sleep": {0},
                "work": {0},
            },
        )

        self.assertEqual(skipped_ids, ["work"])
        self.assertEqual(sorted(filtered_candidates.keys()), ["sleep"])
        self.assertEqual(sorted(filtered_variants.keys()), ["sleep"])
        self.assertEqual(sorted(filtered_rotations.keys()), ["sleep"])
        self.assertEqual(sorted(filtered_outlines.keys()), ["sleep"])
        self.assertEqual(filtered_relation_plan["cluster_relations"], [])
        self.assertEqual(filtered_relation_plan["cluster_affinities"], [])
        self.assertEqual(
            filtered_constraints,
            {"clusters": [{"cluster_id": "sleep"}]},
        )

    def test_relax_relation_plan_for_candidate_search_drops_pairwise_directional_biases(
        self,
    ) -> None:
        relation_plan = {
            "cluster_relations": [{"a": "sleep", "b": "living", "relation": "near"}],
            "cluster_directional_relations": [
                {"a": "sleep", "b": "living", "relation": "face_each_other"}
            ],
            "cluster_orientations": [
                {
                    "cluster_id": "living",
                    "intents": ["face_cluster", "back_to_wall"],
                    "target_cluster_id": "sleep",
                }
            ],
            "object_orientations": [
                {
                    "cluster_id": "living",
                    "object_id": "armchair",
                    "intents": ["face_object", "front_to_open_space"],
                    "target_object_id": "bed",
                }
            ],
        }

        relaxed = _solver._relax_relation_plan_for_candidate_search(relation_plan)

        self.assertEqual(relaxed["cluster_relations"], [])
        self.assertEqual(relaxed["cluster_directional_relations"], [])
        self.assertEqual(
            relaxed["cluster_orientations"],
            [
                {
                    "cluster_id": "living",
                    "intents": ["back_to_wall"],
                    "target_cluster_id": None,
                }
            ],
        )
        self.assertEqual(
            relaxed["object_orientations"],
            [
                {
                    "cluster_id": "living",
                    "object_id": "armchair",
                    "intents": ["front_to_open_space"],
                    "target_object_id": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
