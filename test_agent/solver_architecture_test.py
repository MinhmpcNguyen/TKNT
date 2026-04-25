from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    @staticmethod
    def _make_candidate(
        cluster_id: str,
        *,
        anchor_kind: str = "seed",
        rough_score: int = 100,
        acceptable_valid: bool = True,
        orientation_penalty_mm: int = 0,
        critical_orientation_penalty_mm: int = 0,
        focal_orientation_penalty_mm: int = 0,
        variant_family: str = "base",
        x: int = 0,
        y: int = 0,
        rot: int = 0,
    ) -> Candidate:
        return Candidate(
            cluster_id=cluster_id,
            variant_id=f"{cluster_id}_{variant_family}",
            variant_family=variant_family,
            variant_priority=1.0,
            x=x,
            y=y,
            rot=rot,
            anchor_kind=anchor_kind,
            anchor_priority=1.0,
            stage="seed",
            hard_valid=True,
            acceptable_valid=acceptable_valid,
            rough_score=rough_score,
            macro_penalty_mm=0,
            micro_penalty_mm=0,
            orientation_penalty_mm=orientation_penalty_mm,
            critical_orientation_penalty_mm=critical_orientation_penalty_mm,
            focal_orientation_penalty_mm=focal_orientation_penalty_mm,
            quality_gate_reasons=(),
            hard_error_codes=(),
            state_signature=f"{cluster_id}@{x},{y},{rot}",
        )

    def test_augment_relation_plan_only_promotes_explicit_front_access(self) -> None:
        relation_plan = {"object_orientations": [], "notes": []}
        cluster_constraints = {
            "clusters": [
                {
                    "cluster_id": "seating",
                    "cluster_rules": {
                        "facing": {
                            "sectional_sofa": {"front": "bottom"},
                            "coffee_table": {"front": "top"},
                        },
                        "access_requirements": [
                            {
                                "id": "sectional_sofa",
                                "type": "front_clearance",
                                "required": True,
                            }
                        ],
                    },
                    "hard_constraints": [
                        {
                            "type": "requires_access",
                            "id": "armchair",
                            "mode": "front_clearance",
                        }
                    ],
                }
            ]
        }

        augmented = _solver._augment_relation_plan_from_cluster_constraints(
            relation_plan,
            cluster_constraints,
        )

        object_orientation_ids = sorted(
            str(item.get("object_id") or "")
            for item in augmented["object_orientations"]
            if isinstance(item, dict)
        )
        self.assertEqual(object_orientation_ids, ["armchair", "sectional_sofa"])
        self.assertNotIn("coffee_table", object_orientation_ids)

    def test_ordered_cluster_ids_prefers_enumerator_priority(self) -> None:
        base_candidate = Candidate(
            cluster_id="placeholder",
            variant_id="base",
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
            macro_penalty_mm=0,
            micro_penalty_mm=0,
            orientation_penalty_mm=0,
            critical_orientation_penalty_mm=0,
            focal_orientation_penalty_mm=0,
            quality_gate_reasons=(),
            hard_error_codes=(),
            state_signature="ok",
        )
        candidates_by_cluster = {
            "support": [
                base_candidate.__class__(
                    **{**base_candidate.__dict__, "cluster_id": "support"}
                )
            ],
            "media": [
                base_candidate.__class__(
                    **{**base_candidate.__dict__, "cluster_id": "media"}
                )
            ],
            "seating": [
                base_candidate.__class__(
                    **{**base_candidate.__dict__, "cluster_id": "seating"}
                )
            ],
        }

        ordered = _solver._ordered_cluster_ids(
            candidates_by_cluster,
            cluster_order=["media", "seating"],
        )

        self.assertEqual(ordered, ["media", "seating", "support"])

    def test_select_root_cluster_prefers_primary_wall_anchor(self) -> None:
        candidates_by_cluster = {
            "support": [self._make_candidate("support")],
            "media": [self._make_candidate("media")],
            "seating": [self._make_candidate("seating")],
        }
        relation_plan = {
            "layout_intent_profile": {
                "primary_cluster_id": "media",
                "secondary_cluster_id": "seating",
            },
            "cluster_affinities": [
                {
                    "cluster_id": "media",
                    "prefer": ["wall", "long_wall", "far_from_entry"],
                    "avoid": ["window_blocking", "entry_blocking"],
                },
                {"cluster_id": "seating", "prefer": ["wall"], "avoid": ["center"]},
            ],
            "cluster_orientations": [
                {
                    "cluster_id": "media",
                    "intents": ["back_to_wall", "face_cluster"],
                    "target_cluster_id": "seating",
                    "priority": "high",
                },
                {
                    "cluster_id": "seating",
                    "intents": ["face_cluster"],
                    "target_cluster_id": "media",
                    "priority": "high",
                },
            ],
            "cluster_relations": [
                {
                    "a": "media",
                    "b": "seating",
                    "relation": "near",
                    "priority": "high",
                }
            ],
            "cluster_directional_relations": [
                {
                    "a": "media",
                    "b": "seating",
                    "relation": "face_each_other",
                    "priority": "high",
                }
            ],
        }

        root_cluster_id = _solver._select_root_cluster_id(
            candidates_by_cluster,
            relation_plan=relation_plan,
            cluster_order=["support", "media", "seating"],
        )
        ordered = _solver._root_cluster_first_order(
            candidates_by_cluster,
            relation_plan=relation_plan,
            cluster_order=["support", "media", "seating"],
        )

        self.assertEqual(root_cluster_id, "media")
        self.assertEqual(ordered[:2], ["media", "seating"])

    def test_prepare_round_candidates_wall_pins_root_before_full_search(self) -> None:
        candidates_by_cluster = {
            "media": [
                self._make_candidate(
                    "media",
                    anchor_kind="center_seed",
                    rough_score=800,
                    x=100,
                ),
                self._make_candidate(
                    "media",
                    anchor_kind="rescue_top",
                    rough_score=200,
                    x=200,
                ),
                self._make_candidate(
                    "media",
                    anchor_kind="rescue_left",
                    rough_score=220,
                    x=300,
                ),
                self._make_candidate(
                    "media",
                    anchor_kind="dense_right",
                    rough_score=210,
                    x=400,
                ),
                self._make_candidate(
                    "media",
                    anchor_kind="rescue_bottom",
                    rough_score=190,
                    x=500,
                ),
            ],
            "seating": [self._make_candidate("seating", anchor_kind="center_seed")],
        }
        relation_plan = {
            "layout_intent_profile": {"primary_cluster_id": "media"},
            "cluster_affinities": [
                {
                    "cluster_id": "media",
                    "prefer": ["wall", "long_wall"],
                    "avoid": ["window_blocking", "entry_blocking"],
                }
            ],
            "cluster_orientations": [
                {"cluster_id": "media", "intents": ["back_to_wall"], "priority": "high"}
            ],
        }

        prioritized, ordered, _bonus_by_key, debug = _solver._prepare_round_candidates(
            candidates_by_cluster,
            relation_plan=relation_plan,
            cluster_order=["media", "seating"],
            wall_pin_focus=True,
        )

        self.assertEqual(ordered[0], "media")
        self.assertTrue(debug["used_wall_pin_focus"])
        self.assertTrue(prioritized["media"])
        self.assertTrue(
            all(
                _solver._anchor_kind_is_wall_pinned(candidate.anchor_kind)
                for candidate in prioritized["media"]
            )
        )

    def test_prepare_round_candidates_prefers_orientation_sound_wall_candidates(
        self,
    ) -> None:
        candidates_by_cluster = {
            "seating": [
                self._make_candidate(
                    "seating",
                    anchor_kind="wall_top",
                    rough_score=1400,
                    acceptable_valid=False,
                    critical_orientation_penalty_mm=420,
                    focal_orientation_penalty_mm=260,
                    orientation_penalty_mm=540,
                    x=100,
                ),
                self._make_candidate(
                    "seating",
                    anchor_kind="wall_bottom",
                    rough_score=1020,
                    acceptable_valid=True,
                    critical_orientation_penalty_mm=18,
                    focal_orientation_penalty_mm=14,
                    orientation_penalty_mm=36,
                    x=200,
                ),
                self._make_candidate(
                    "seating",
                    anchor_kind="wall_left",
                    rough_score=990,
                    acceptable_valid=True,
                    critical_orientation_penalty_mm=24,
                    focal_orientation_penalty_mm=16,
                    orientation_penalty_mm=44,
                    x=300,
                ),
                self._make_candidate(
                    "seating",
                    anchor_kind="wall_right",
                    rough_score=970,
                    acceptable_valid=True,
                    critical_orientation_penalty_mm=28,
                    focal_orientation_penalty_mm=18,
                    orientation_penalty_mm=52,
                    x=350,
                ),
                self._make_candidate(
                    "seating",
                    anchor_kind="center_seed",
                    rough_score=2200,
                    acceptable_valid=True,
                    x=450,
                ),
            ],
            "media": [self._make_candidate("media", anchor_kind="wall_right")],
        }
        relation_plan = {
            "layout_intent_profile": {
                "primary_cluster_id": "seating",
                "secondary_cluster_id": "media",
            },
            "cluster_affinities": [
                {"cluster_id": "seating", "prefer": ["wall", "long_wall"]},
                {"cluster_id": "media", "prefer": ["wall"]},
            ],
            "cluster_orientations": [
                {
                    "cluster_id": "seating",
                    "intents": ["back_to_wall", "face_cluster"],
                    "target_cluster_id": "media",
                    "priority": "high",
                }
            ],
        }

        prioritized, ordered, _bonus_by_key, debug = _solver._prepare_round_candidates(
            candidates_by_cluster,
            relation_plan=relation_plan,
            cluster_order=["seating", "media"],
            wall_pin_focus=True,
        )

        self.assertEqual(ordered[0], "seating")
        self.assertTrue(debug["used_wall_pin_focus"])
        self.assertTrue(debug["used_hard_wall_contract"])
        self.assertEqual(prioritized["seating"][0].anchor_kind, "wall_bottom")
        self.assertNotIn(
            "wall_top",
            [candidate.anchor_kind for candidate in prioritized["seating"]],
        )

    def test_relax_relation_plan_preserves_viewing_contracts(self) -> None:
        relation_plan = {
            "layout_intent_profile": {
                "focus_mode": "viewing",
                "primary_cluster_id": "seating",
                "secondary_cluster_id": "media",
            },
            "cluster_relations": [
                {"a": "seating", "b": "media", "relation": "near"},
                {"a": "seating", "b": "support", "relation": "near"},
            ],
            "cluster_directional_relations": [
                {"a": "seating", "b": "media", "relation": "face_each_other"},
                {"a": "seating", "b": "support", "relation": "face_same_direction"},
            ],
            "cluster_orientations": [
                {
                    "cluster_id": "seating",
                    "intents": ["back_to_wall", "face_cluster"],
                    "target_cluster_id": "media",
                },
                {
                    "cluster_id": "support",
                    "intents": ["face_cluster"],
                    "target_cluster_id": "seating",
                },
            ],
            "object_orientations": [
                {
                    "cluster_id": "seating",
                    "object_id": "sofa",
                    "intents": ["face_object"],
                    "target_object_id": "tv_console",
                    "target_object_cluster_id": "media",
                },
                {
                    "cluster_id": "support",
                    "object_id": "desk",
                    "intents": ["face_object"],
                    "target_object_id": "door",
                    "target_object_cluster_id": "entry",
                },
            ],
        }

        relaxed = _solver._relax_relation_plan_for_candidate_search(relation_plan)

        self.assertEqual(len(relaxed["cluster_relations"]), 1)
        self.assertEqual(relaxed["cluster_relations"][0]["b"], "media")
        self.assertEqual(len(relaxed["cluster_directional_relations"]), 1)
        self.assertEqual(
            relaxed["cluster_directional_relations"][0]["relation"], "face_each_other"
        )
        seating_orientation = next(
            row
            for row in relaxed["cluster_orientations"]
            if row.get("cluster_id") == "seating"
        )
        self.assertIn("face_cluster", seating_orientation["intents"])
        self.assertFalse(
            any(
                row.get("cluster_id") == "support"
                and "face_cluster" in (row.get("intents") or [])
                for row in relaxed["cluster_orientations"]
            )
        )
        self.assertEqual(len(relaxed["object_orientations"]), 1)
        self.assertEqual(relaxed["object_orientations"][0]["cluster_id"], "seating")

    def test_wall_pin_focus_next_action_never_drops_mandatory_focus(self) -> None:
        self.assertEqual(
            _solver._wall_pin_focus_next_action(
                round_strategy={"used_wall_pin_focus": True},
            ),
            "refine_with_focus",
        )
        self.assertEqual(
            _solver._wall_pin_focus_next_action(
                round_strategy={"used_wall_pin_focus": True},
            ),
            "refine_with_focus",
        )
        self.assertEqual(
            _solver._wall_pin_focus_next_action(
                round_strategy={"used_wall_pin_focus": False},
            ),
            "none",
        )

    def test_verify_rank_key_prefers_macro_health_before_micro(self) -> None:
        macro_bad = {
            "hard_valid": True,
            "acceptable_valid": False,
            "complete": True,
            "quality": {
                "layout_score": 900,
                "macro_penalty_mm": 700,
                "micro_penalty_mm": 40,
                "critical_orientation_penalty_mm": 300,
                "focal_pair_penalty_mm": 200,
            },
        }
        micro_bad = {
            "hard_valid": True,
            "acceptable_valid": False,
            "complete": True,
            "quality": {
                "layout_score": 800,
                "macro_penalty_mm": 120,
                "micro_penalty_mm": 600,
                "critical_orientation_penalty_mm": 0,
                "focal_pair_penalty_mm": 0,
            },
        }

        self.assertGreater(
            _solver._verify_rank_key(micro_bad),
            _solver._verify_rank_key(macro_bad),
        )
        self.assertGreater(
            _solver._verify_search_score(micro_bad),
            _solver._verify_search_score(macro_bad),
        )

    def test_build_placer_seed_blocks_micro_repair_until_macro_is_ready(self) -> None:
        verify = {
            "hard_valid": True,
            "acceptable_valid": False,
            "complete": True,
            "quality": {
                "layout_score": 1200,
                "macro_penalty_mm": 900,
                "micro_penalty_mm": 120,
            },
            "quality_gate": {
                "reasons": ["critical_orientation_penalty_too_high"],
            },
            "repair_guidance": {"prioritized_clusters": [], "conflict_sets": []},
            "state_signature": "seed",
        }

        seed = _solver._build_placer_seed(
            tools=SimpleNamespace(
                MaterializeVariantizedClusters=lambda **kwargs: {
                    "result": "OK",
                    "clusters_outlines": kwargs.get("clusters_outlines"),
                }
            ),
            clusters_outlines={},
            relation_plan_used={},
            cluster_constraints_used=None,
            grid_mm=50,
            cluster_transforms=[
                {"cluster_id": "media", "x": 0, "y": 0, "rot": 0},
            ],
            selected_variants=[
                {"cluster_id": "media", "variant_id": "media_base"},
            ],
            verify=verify,
            solver_status="UNSAT",
            candidate_counts={"media": 4},
        )

        self.assertEqual(seed["phase"], "macro_blocked")
        self.assertFalse(seed["ready"])
        self.assertFalse(seed["needed"])
        self.assertFalse(seed["seed_metrics"]["macro_ready"])

    def test_publishable_gate_rejects_hard_valid_macro_bad_solution(self) -> None:
        candidate = self._make_candidate(
            "main_seating",
            variant_family="open_center",
        )
        verify = {
            "hard_valid": True,
            "acceptable_valid": False,
            "complete": True,
            "quality": {
                "layout_score": 100,
                "circulation_penalty_mm": 500,
                "critical_orientation_penalty_mm": 900,
                "focal_pair_penalty_mm": 700,
                "circulation_debug": [
                    {"kind": "center_lane", "penalty_mm": 320},
                ],
            },
            "quality_gate": {
                "reasons": [
                    "critical_orientation_penalty_too_high",
                    "focal_pair_penalty_too_high",
                    "critical_item_penalty_too_high",
                ]
            },
        }
        relation_plan = {
            "macro_concept": {
                "concept_family": "open_center",
                "topology_policy": {
                    "reserve_center_degree": "very_high",
                    "preserve_entry_landing": True,
                    "preserve_primary_corridor": True,
                },
                "cluster_zone_plan": [
                    {
                        "cluster_id": "main_seating",
                        "priority": "core",
                        "center_usage": "open_reserved",
                    }
                ],
            }
        }
        record = {
            "hard_valid": True,
            "selected_variants": [
                {
                    "cluster_id": "main_seating",
                    "variant_id": "main_seating__open_center__01",
                }
            ],
            "diagnostics": _solver._solution_diagnostics(
                verify=verify,
                relation_plan=relation_plan,
                chosen={"main_seating": candidate},
            ),
            "verify_summary": {
                "quality_gate_reasons": verify["quality_gate"]["reasons"],
            },
            "_verify": verify,
        }

        reasons = _solver._solution_publishability_violations(
            record,
            relation_plan=relation_plan,
        )

        self.assertIn("critical_orientation_penalty_too_high", reasons)
        self.assertIn("focal_pair_penalty_too_high", reasons)
        self.assertIn("critical_item_penalty_too_high", reasons)
        self.assertIn("circulation_clear_false", reasons)
        self.assertIn("center_openness_preserved_false", reasons)
        self.assertFalse(
            _solver._is_publishable_solution(record, relation_plan=relation_plan)
        )

    def test_concept_variant_policy_filters_fallback_and_role_mismatch(self) -> None:
        relation_plan = {
            "macro_concept": {
                "concept_family": "open_center",
                "cluster_zone_plan": [
                    {
                        "cluster_id": "main_seating",
                        "priority": "core",
                        "role_kind": "social_anchor",
                    },
                    {
                        "cluster_id": "media",
                        "priority": "core",
                        "role_kind": "media",
                    },
                ],
                "allowed_variant_families_by_cluster": {
                    "main_seating": ["open_center", "conversation_facing"],
                    "media": ["media_facing"],
                },
            }
        }
        candidates_by_cluster = {
            "main_seating": [
                self._make_candidate("main_seating", variant_family="fallback_generic"),
                self._make_candidate("main_seating", variant_family="open_center"),
            ],
            "media": [
                self._make_candidate("media", variant_family="conversation_facing"),
                self._make_candidate("media", variant_family="media_facing"),
            ],
        }

        filtered = _solver._filter_candidates_by_concept_variant_policy(
            candidates_by_cluster,
            relation_plan,
        )

        self.assertEqual(
            [candidate.variant_family for candidate in filtered["main_seating"]],
            ["open_center"],
        )
        self.assertEqual(
            [candidate.variant_family for candidate in filtered["media"]],
            ["media_facing"],
        )

    def test_cluster_role_kind_infers_sleep_without_zone_plan(self) -> None:
        relation_plan = {"macro_concept": {"concept_family": "zoned"}}

        self.assertEqual(
            _solver._cluster_role_kind(relation_plan, "sleep_area"),
            "sleep",
        )
        self.assertEqual(
            _solver._cluster_priority_kind(relation_plan, "sleep_area"),
            "core",
        )

    def test_concept_variant_policy_prefers_sleep_families_for_bedroom_cluster(
        self,
    ) -> None:
        relation_plan = {
            "macro_concept": {
                "concept_family": "daylight_oriented",
                "cluster_zone_plan": [
                    {
                        "cluster_id": "sleep_area",
                        "priority": "core",
                        "role_kind": "sleep",
                    }
                ],
                "allowed_variant_families_by_cluster": {
                    "sleep_area": [
                        "headboard_wall_balanced",
                        "bed_plus_window_side_bench",
                    ]
                },
            }
        }
        candidates_by_cluster = {
            "sleep_area": [
                self._make_candidate(
                    "sleep_area",
                    variant_family="conversation_facing",
                ),
                self._make_candidate(
                    "sleep_area",
                    variant_family="headboard_wall_balanced",
                ),
                self._make_candidate(
                    "sleep_area",
                    variant_family="bed_plus_window_side_bench",
                ),
            ]
        }

        filtered = _solver._filter_candidates_by_concept_variant_policy(
            candidates_by_cluster,
            relation_plan,
        )

        self.assertEqual(
            [candidate.variant_family for candidate in filtered["sleep_area"]],
            ["headboard_wall_balanced", "bed_plus_window_side_bench"],
        )
        self.assertGreater(
            _solver._candidate_variant_family_bonus(
                filtered["sleep_area"][1],
                relation_plan=relation_plan,
            ),
            _solver._candidate_variant_family_bonus(
                filtered["sleep_area"][0],
                relation_plan=relation_plan,
            ),
        )

    def test_primary_seed_problem_clusters_limits_to_top_two(self) -> None:
        problem_clusters = _solver._primary_seed_problem_clusters(
            ["media_zone", "seating_zone", "utility_storage"],
            cluster_ids=["media_zone", "seating_zone", "utility_storage"],
        )

        self.assertEqual(problem_clusters, ["media_zone", "seating_zone"])

    def test_variant_bundle_adapter_keeps_plural_hulls_and_rects(self) -> None:
        converted = _solver._cluster_payload_from_variant_bundle(
            {
                "cluster_id": "main_seating",
                "variant_bundle": [
                    {
                        "variant_id": "main_seating__open_center",
                        "variant_family": "open_center",
                        "local_placements": [{"id": "sofa", "x": 0, "y": 0, "rot": 0}],
                        "interaction_placements": [
                            {"id": "sofa", "x": 0, "y": 0, "w": 2200, "h": 900}
                        ],
                        "tight_hull_polygon_mm": [],
                        "tight_hull_polygons_mm": [
                            [
                                {"x": 0, "y": 0},
                                {"x": 2200, "y": 0},
                                {"x": 2200, "y": 900},
                                {"x": 0, "y": 900},
                            ]
                        ],
                    }
                ],
            }
        )

        self.assertIsNotNone(converted)
        assert converted is not None
        _, payload = converted
        footprint = payload["cluster_footprint"]
        self.assertEqual(len(footprint["rects"]), 1)
        self.assertEqual(len(footprint["outline_polygons_ccw"]), 1)
        self.assertEqual(footprint["local_bbox"]["max_x"], 2200)

    def test_primary_seed_candidate_growth_is_tightly_bounded(self) -> None:
        self.assertEqual(_solver._round_candidate_growth_limit(28), 12)
        self.assertEqual(_solver._expanded_candidate_limit(28), 36)

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
            macro_penalty_mm=0,
            micro_penalty_mm=0,
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

        self.assertEqual(
            relaxed["cluster_relations"],
            [{"a": "sleep", "b": "living", "relation": "near"}],
        )
        self.assertEqual(
            relaxed["cluster_directional_relations"],
            [{"a": "sleep", "b": "living", "relation": "face_each_other"}],
        )
        self.assertEqual(
            relaxed["cluster_orientations"],
            [
                {
                    "cluster_id": "living",
                    "intents": ["face_cluster", "back_to_wall"],
                    "target_cluster_id": "sleep",
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
