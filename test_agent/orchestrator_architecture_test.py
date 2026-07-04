from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_module(name: str, **attributes: object) -> None:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


_STUBBED_MODULE_NAMES = [
    "agent.seed_relation_planner",
    "agent.room_interpreter",
    "agent.seed_concept_generator",
    "agent.solver.solver",
    "agent.stylist",
    "agent.tier_count_director",
    "object_program.builder",
    "layout.grid_policy",
    "pipeline.dropped_inventory",
    "pipeline.layout_variants",
    "pipeline.manual_placements",
]


_stub_module("agent.seed_relation_planner", SeedRelationPlanner=object)
_stub_module("agent.room_interpreter", RoomInterpreter=object)
_stub_module(
    "agent.seed_concept_generator",
    SeedConceptGenerator=object,
    solver_plan_from_concept=lambda **kwargs: {},
)
_stub_module("agent.solver.solver", MacroClusterSolver=object)
_stub_module(
    "agent.stylist",
    Stylist=object,
    stylist_model_name_for_variant=lambda variant_index: "gpt-5.4-mini",
)
_stub_module("agent.tier_count_director", TierCountDirector=object)
_stub_module(
    "object_program.builder",
    build_solver_object_program=lambda cluster_output, tier_output: {
        "clusters": [
            {
                "cluster_id": "living",
                "members": ["sofa"],
                "decisions": [
                    {
                        "object_type": "sofa",
                        "quantity": 1,
                        "size_tier": "M",
                    }
                ],
            }
        ]
    },
)
_stub_module("layout.grid_policy", GLOBAL_LAYOUT_GRID_MM=50)
_stub_module(
    "pipeline.dropped_inventory",
    controlled_accessory_refill=lambda **kwargs: (
        kwargs.get("absolute_layout") or {},
        {"refill_count": 0},
    ),
    dropped_inventory_payload=lambda value: value,
)
_stub_module(
    "pipeline.layout_variants",
    build_final_gallery_selection_summary=lambda **kwargs: {
        "selected_count": len(kwargs.get("selected") or []),
    },
    core_anchor_layout_signature=lambda _layout: "",
    material_macro_layout_signature=lambda _layout: "",
    select_distinct_final_gallery_candidates=lambda **kwargs: kwargs.get("candidates")
    or [],
)
_stub_module(
    "pipeline.manual_placements",
    apply_manual_placements_to_tier_output=lambda layout, *_args, **_kwargs: layout,
    append_manual_placements_guidance=lambda text, *_args, **_kwargs: text,
    augment_seed_relation_plan_with_manual_placements=lambda payload,
    *_args,
    **_kwargs: payload,
    merge_manual_placements_into_absolute_layout=lambda layout,
    *_args,
    **_kwargs: layout,
    merge_manual_placements_into_styled_output=lambda layout, *_args, **_kwargs: layout,
    prepare_input_payload_for_manual_placements=lambda payload,
    *_args,
    **_kwargs: payload,
)


orchestrator = _load_module(
    "pipeline.orchestrator_for_architecture_test",
    "pipeline/orchestrator.py",
)
request_non_functional_refill = _load_module(
    "pipeline.request_non_functional_refill_for_architecture_test",
    "pipeline/request_non_functional_refill.py",
)

for _module_name in _STUBBED_MODULE_NAMES:
    sys.modules.pop(_module_name, None)


def _repair_relation_plan() -> dict[str, object]:
    return {
        "notes": [],
        "anchor_region_preferences_by_cluster": {
            "sleep_core": {
                "zone_assignment": "right_wall_zone",
                "preferred_region_ids": [
                    "right_wall_zone",
                    "quiet_private_deep_zone",
                    "bottom_wall_zone",
                ],
                "forbidden_region_ids": ["door_1_entry_clearance"],
                "preferred_wall_side": "right_wall",
                "anchor_strength": "strong",
                "center_usage": "primary",
                "placement_bias": "wall_backed",
                "placement_behavior": {
                    "wall_backing": "required",
                    "pose_flexibility": "low",
                },
            },
            "work_study": {
                "zone_assignment": "right_wall_zone",
                "preferred_region_ids": [
                    "right_wall_zone",
                    "edge_loading_zone",
                    "bottom_wall_zone",
                    "window_side_zone",
                ],
                "forbidden_region_ids": [
                    "door_1_entry_clearance",
                    "entry_to_center_corridor",
                ],
                "preferred_wall_side": "right_wall",
                "anchor_strength": "strong",
                "center_usage": "none",
                "placement_bias": "wall_backed",
                "placement_behavior": {
                    "wall_backing": "required",
                    "pose_flexibility": "low",
                },
            },
        },
        "macro_concept": {
            "cluster_zone_plan": [
                {"cluster_id": "sleep_core", "zone_assignment": "right_wall_zone"},
                {"cluster_id": "work_study", "zone_assignment": "right_wall_zone"},
            ],
            "critical_object_orientations": [
                {
                    "cluster_id": "work_study",
                    "object_id": "desk",
                    "target_region": "right_wall_zone",
                }
            ],
        },
        "object_orientations": [
            {
                "cluster_id": "work_study",
                "object_id": "desk",
                "target_region": "right_wall_zone",
            }
        ],
        "macro_constraints": {
            "protected_topology": [{"region": "door_1_entry_clearance"}],
            "anchor_region_constraints": [
                {
                    "cluster_id": "work_study",
                    "zone_assignment": "right_wall_zone",
                    "forbidden_region_ids": [
                        "door_1_entry_clearance",
                        "entry_to_center_corridor",
                    ],
                }
            ],
        },
        "object_solver_policy": {
            "protected_ids_by_cluster": {"work_study": ["desk", "chair"]},
            "droppable_ids_by_cluster": {"work_study": []},
        },
    }


def _repair_merged_output() -> dict[str, object]:
    return {
        "clusters": [
            {
                "cluster_id": "work_study",
                "members": ["desk", "chair"],
                "anchors": ["desk"],
                "cluster_rules": {
                    "anchor_first_policy": {
                        "dominant_anchor_candidates": ["desk"],
                        "protected_ids": ["desk", "chair"],
                        "droppable_ids": [],
                    },
                    "tier_count_hints": {
                        "object_hints": [
                            {"object_type": "desk", "min_keep": 1},
                            {"object_type": "chair", "min_keep": 0},
                        ]
                    },
                },
            }
        ]
    }


def _unsat_solver_output(
    *,
    reasons: list[str],
    offenders: list[str],
    raw_solution_count: int,
    anchor_candidate_counts: dict[str, int] | None = None,
    failure_kind: str = "",
    pairwise_incompatible_anchor_pairs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    anchor_counts = anchor_candidate_counts or {"sleep_core": 6, "work_study": 36}
    solver_debug: dict[str, object] = {
        "raw_solution_count": raw_solution_count,
        "anchor_candidate_counts": anchor_counts,
    }
    if failure_kind:
        solver_debug["failure_kind"] = failure_kind
    if pairwise_incompatible_anchor_pairs is not None:
        solver_debug["pairwise_incompatible_anchor_pairs"] = (
            pairwise_incompatible_anchor_pairs
        )
    return {
        "status": "UNSAT",
        "failure_kind": failure_kind,
        "offending_clusters": offenders,
        "solver_debug": solver_debug,
        "best_partial": {
            "verify_summary": {
                "quality_gate_reasons": reasons,
                "offending_clusters": offenders,
                "blocking_functional_geometry_issues": [
                    {"cluster_id": offender, "reason": reasons[0]}
                    for offender in offenders
                ],
            }
        },
    }


def _ok_object_solver_output() -> dict[str, object]:
    return {
        "status": "OK",
        "hard_valid": True,
        "geometry_valid": True,
        "complete": True,
        "gallery_eligible": True,
        "coverage_ratio": 1.0,
        "absolute_layout": {
            "status": "OK",
            "clusters": [{"cluster_id": "work_study"}],
            "objects": [],
        },
        "verify_summary": {
            "layout_score": 100,
            "missing_cluster_ids": [],
        },
    }


def _front_access_unsat_solver_output() -> dict[str, object]:
    return {
        "status": "UNSAT",
        "solver_debug": {
            "raw_solution_count": 24,
            "anchor_candidate_counts": {"sleep_core": 5, "storage_core": 22},
        },
        "solver_degradation_report": {
            "raw_solution_count": 24,
            "top_quality_gate_reasons": [
                {"reason": "front_access_zone_blocked", "count": 24}
            ],
        },
        "best_partial": {
            "verify_summary": {
                "quality_gate_reasons": ["front_access_zone_blocked"],
                "offending_clusters": ["sleep_core"],
                "blocking_functional_geometry_issues": [
                    {
                        "cluster_id": "sleep_core",
                        "object_id": "bed",
                        "blocked_cluster_id": "storage_core",
                        "blocked_object_id": "wardrobe",
                        "reason": "front_access_zone_blocked",
                        "overlap_ratio": 0.341,
                        "overlap_area_mm2": 1100500,
                    }
                ],
            }
        },
    }


def _daylight_quality_unsat_solver_output() -> dict[str, object]:
    return {
        "status": "UNSAT",
        "solver_debug": {
            "raw_solution_count": 12,
            "anchor_candidate_counts": {"storage_core": 8},
        },
        "solver_degradation_report": {
            "raw_solution_count": 12,
            "top_quality_gate_reasons": [
                {"reason": "blocking_daylight_quality_overlap", "count": 12}
            ],
        },
        "best_partial": {
            "verify_summary": {
                "quality_gate_reasons": ["blocking_daylight_quality_overlap"],
                "offending_clusters": ["storage_core"],
                "blocking_forbidden_region_issues": [
                    {
                        "cluster_id": "storage_core",
                        "object_id": "wardrobe",
                        "region_id": "window_1_daylight",
                        "zone_type": "daylight_quality_blocking",
                        "overlap_ratio": 0.42,
                        "overlap_area_mm2": 420000,
                    }
                ],
            }
        },
    }


def _front_access_unsat_with_storage_wall_option() -> dict[str, object]:
    output = _front_access_unsat_solver_output()
    solver_debug = output["solver_debug"]
    if not isinstance(solver_debug, dict):
        raise AssertionError("solver_debug fixture must be a dict")
    solver_debug["anchor_wall_options_by_cluster"] = {
        "storage_core": {
            "top": {
                "valid_candidate_count": 3,
                "front_inward_candidate_count": 2,
                "wall_parallel_candidate_count": 2,
                "best_score": 18250.0,
                "best_candidate": {
                    "front_inward_match": True,
                    "long_axis_wall_parallel": True,
                    "candidate_score": 18250.0,
                },
            },
            "right": {
                "valid_candidate_count": 0,
                "front_inward_candidate_count": 0,
                "wall_parallel_candidate_count": 0,
                "reject_reason_counts": {"forbidden_overlap:entry_clearance": 4},
            },
        }
    }
    return output


class _FakeObjectLevelSolver:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def generate_object_layout(self, **kwargs):
        self.calls.append(kwargs)
        if self.outputs:
            return self.outputs.pop(0)
        return _unsat_solver_output(
            reasons=["view_corridor_blocked"],
            offenders=["work_study"],
            raw_solution_count=192,
        )


class OrchestratorArchitectureTest(unittest.TestCase):
    def test_partial_layout_variants_publish_every_completed_candidate(self) -> None:
        original_selector = orchestrator.select_distinct_final_gallery_candidates
        orchestrator.select_distinct_final_gallery_candidates = (
            lambda *, candidates, max_variants: candidates[:1]
        )
        try:
            with TemporaryDirectory() as tmp:
                paths = orchestrator.CasePaths(
                    case_id="case_partial_variants",
                    root=Path(tmp),
                )
                candidates = [
                    {
                        "absolute_layout": {"objects": [{"object_id": "sofa"}]},
                        "styled_result": {"objects": [{"instance_id": "sofa"}]},
                        "source": "candidate_1",
                        "hard_valid": True,
                        "complete": True,
                        "gallery_eligible": True,
                    },
                    {
                        "absolute_layout": {"objects": [{"object_id": "desk"}]},
                        "styled_result": {"objects": [{"instance_id": "desk"}]},
                        "source": "candidate_2",
                        "hard_valid": True,
                        "complete": True,
                        "gallery_eligible": True,
                    },
                ]

                orchestrator._write_partial_layout_variants_artifact(
                    paths=paths,
                    final_candidates=candidates,
                    target_final_count=3,
                    ablation_mode="full",
                )

                payload = json.loads(paths.layout_variants.read_text())

            self.assertTrue(payload["is_partial"])
            self.assertEqual(payload["ready_count"], 2)
            self.assertEqual(
                [variant["source"] for variant in payload["variants"]],
                ["candidate_1", "candidate_2"],
            )
            self.assertEqual(payload["selection_summary"], {"selected_count": 1})
        finally:
            orchestrator.select_distinct_final_gallery_candidates = original_selector

    def test_layout_variants_copy_orientation_into_styled_output(self) -> None:
        payload = orchestrator._build_layout_variants_payload_from_final_candidates(
            [
                {
                    "absolute_layout": {
                        "objects": [
                            {
                                "object_id": "bed",
                                "rotation_ccw": 270,
                                "front_world": {"dx": -1, "dy": 0},
                                "front_side_world": "left",
                            }
                        ]
                    },
                    "styled_result": {
                        "objects": [
                            {
                                "instance_id": "bed",
                                "object_type": "bed",
                            }
                        ]
                    },
                }
            ],
            status="OK",
        )

        bed = payload["variants"][0]["styled_result"]["objects"][0]
        self.assertEqual(bed["rotation_ccw"], 270)
        self.assertEqual(bed["front_world"], {"dx": -1, "dy": 0})
        self.assertEqual(bed["front_side_world"], "left")

    def test_requested_non_functional_specs_use_stylist_place_on_methods(
        self,
    ) -> None:
        allowed_methods = {"on_top", "hang_on", "lean_on", "floor"}

        for (
            object_type,
            spec,
        ) in request_non_functional_refill._REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS.items():
            place_on = spec.get("place_on") if isinstance(spec, dict) else None
            if not isinstance(place_on, dict):
                continue
            method = str(place_on.get("method") or "")
            if not method:
                continue

            self.assertIn(
                method,
                allowed_methods,
                f"{object_type} uses unsupported place_on.method={method}",
            )

    def test_select_final_styling_candidates_caps_to_selected_top_five(self) -> None:
        candidates = [
            {
                "source": f"candidate_{index}",
                "absolute_layout": {"objects": [], "room": {}, "status": "OK"},
            }
            for index in range(1, 10)
        ]

        original_selector = orchestrator.select_distinct_final_gallery_candidates
        orchestrator.select_distinct_final_gallery_candidates = lambda **kwargs: list(
            (kwargs.get("candidates") or [])[:5]
        )
        try:
            selected = orchestrator._select_final_styling_candidates(
                candidates,
                max_variants=5,
            )
        finally:
            orchestrator.select_distinct_final_gallery_candidates = original_selector

        self.assertEqual(len(selected), 5)
        self.assertEqual(
            [item["source"] for item in selected],
            [f"candidate_{index}" for index in range(1, 6)],
        )

    def test_core_anchor_diversity_repair_retargets_secondary_first(self) -> None:
        concept = {
            "concept_id": "concept_04",
            "concept_family": "edge_weighted",
            "cluster_zone_plan": [
                {
                    "cluster_id": "sleep_core",
                    "zone_assignment": "quiet_private_deep_zone",
                    "committed_region_ids": ["quiet_private_deep_zone"],
                    "preferred_region_ids": ["quiet_private_deep_zone"],
                    "scenario_role": "primary",
                    "anchor_strength": "strong",
                },
                {
                    "cluster_id": "storage_core",
                    "zone_assignment": "right_wall_zone",
                    "committed_region_ids": ["right_wall_zone"],
                    "preferred_region_ids": [
                        "right_wall_zone",
                        "bottom_wall_zone",
                        "left_wall_zone",
                    ],
                    "scenario_role": "support",
                    "anchor_strength": "strong",
                },
                {
                    "cluster_id": "media_optional",
                    "zone_assignment": "right_wall_zone",
                    "committed_region_ids": ["right_wall_zone"],
                    "preferred_region_ids": ["right_wall_zone", "top_wall_zone"],
                    "scenario_role": "secondary",
                    "anchor_strength": "strong",
                },
            ],
        }

        repairs = orchestrator._build_core_anchor_diversity_repair_concepts(
            concepts=[concept],
            solved_bundles=[
                {
                    "planned_material_macro_signature": (
                        "media_optional:right_wall_zone|"
                        "sleep_core:quiet_private_deep_zone|"
                        "storage_core:right_wall_zone"
                    )
                }
            ],
            selection_summary={
                "returned_count": 5,
                "core_anchor_distinct_candidate_count": 1,
            },
            target_final_count=5,
        )

        self.assertTrue(repairs)
        rows = {row["cluster_id"]: row for row in repairs[0]["cluster_zone_plan"]}
        self.assertEqual(
            rows["sleep_core"]["zone_assignment"],
            "quiet_private_deep_zone",
        )
        self.assertNotEqual(
            rows["storage_core"]["zone_assignment"],
            "right_wall_zone",
        )
        self.assertTrue(repairs[0]["core_anchor_diversity_repair_applied"])

    def test_select_final_styling_candidates_returns_empty_when_selector_empty(
        self,
    ) -> None:
        candidates = [
            {
                "source": f"candidate_{index}",
                "absolute_layout": {"objects": [], "room": {}, "status": "OK"},
            }
            for index in range(1, 10)
        ]

        original_selector = orchestrator.select_distinct_final_gallery_candidates
        orchestrator.select_distinct_final_gallery_candidates = lambda **_kwargs: []
        try:
            selected = orchestrator._select_final_styling_candidates(
                candidates,
                max_variants=5,
            )
        finally:
            orchestrator.select_distinct_final_gallery_candidates = original_selector

        self.assertEqual(selected, [])

    def test_run_case_uses_deterministic_seed_generation(self) -> None:
        class _FakeModelOutput:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def model_dump(self) -> dict[str, object]:
                return deepcopy(self._payload)

        class _FakeRoomInterpreter:
            def generate(self, *_args: object, **_kwargs: object) -> _FakeModelOutput:
                return _FakeModelOutput(
                    {
                        "room": {
                            "room_id": "room_1",
                            "polygon_ccw": [
                                {"x": 0, "y": 0},
                                {"x": 6000, "y": 0},
                                {"x": 6000, "y": 4000},
                                {"x": 0, "y": 4000},
                            ],
                        },
                        "meta": {"room_type": "bedroom"},
                        "notes": ["Place a bedroom with media and study furniture."],
                    }
                )

        class _FakeSeedRelationPlanner:
            def generate(self, *_args: object, **_kwargs: object) -> _FakeModelOutput:
                return _FakeModelOutput({"clusters": []})

        class _FakeTierCountDirector:
            def generate(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {"decisions": []}

        class _FakeStylist:
            def compile_style_policy(
                self, *_args: object, **_kwargs: object
            ) -> dict[str, object]:
                return {}

        class _FakeSeedConceptGenerator:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                concept_generators.append(self)

            def generate_bundle(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(dict(kwargs))
                return {
                    "status": "OK",
                    "room_type": "bedroom",
                    "controller": "deterministic_seed_concept_generator",
                    "settings": {},
                    "concepts": [
                        {
                            "concept_id": "concept_01",
                            "concept_family": "focal_axis",
                            "cluster_zone_plan": [],
                        }
                    ],
                }

        concept_generators: list[_FakeSeedConceptGenerator] = []
        replacements = {
            "RoomInterpreter": _FakeRoomInterpreter,
            "SeedRelationPlanner": _FakeSeedRelationPlanner,
            "TierCountDirector": _FakeTierCountDirector,
            "SeedConceptGenerator": _FakeSeedConceptGenerator,
            "Stylist": _FakeStylist,
            "_build_runtime_solver": lambda: object(),
            "prepare_input_payload_for_manual_placements": lambda payload: (
                payload,
                None,
            ),
            "build_solver_object_program": lambda _cluster_output, _tier_output: {
                "clusters": []
            },
            "_solve_seed_concept_pool": lambda **_kwargs: [],
        }
        originals = {name: getattr(orchestrator, name) for name in replacements}
        for name, replacement in replacements.items():
            setattr(orchestrator, name, replacement)
        try:
            with TemporaryDirectory() as tmp_dir:
                result = orchestrator.run_case(
                    input_payload={
                        "user_input": {
                            "room_type": "bedroom",
                            "description": "bed, desk, chair, wardrobe, tv console",
                        }
                    },
                    user_id="user_1",
                    cases_root=tmp_dir,
                    case_id="case_1",
                )
        finally:
            for name, original in originals.items():
                setattr(orchestrator, name, original)

        self.assertEqual(
            result["error"],
            "object-level solver could not produce any feasible anchor-first layouts",
        )
        self.assertEqual(len(concept_generators), 1)
        generator = concept_generators[0]
        calls = generator.calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["target_count"], 5)
        self.assertNotIn("description", calls[0])

    def test_semantic_cluster_member_dedupe_drops_duplicate_support_cluster(
        self,
    ) -> None:
        payload = {
            "clusters": [
                {
                    "cluster_id": "main_seating",
                    "priority": "core",
                    "layout_role": "primary",
                    "members": ["sofa", "floor_lamp"],
                    "anchors": ["sofa"],
                },
                {
                    "cluster_id": "lighting_support",
                    "priority": "support",
                    "layout_role": "secondary",
                    "members": ["floor_lamp"],
                    "anchors": ["floor_lamp"],
                },
            ],
            "semantic_layout_program": {
                "active_clusters": [
                    {
                        "cluster_id": "main_seating",
                        "required_bundles": [
                            {
                                "bundle_id": "main",
                                "objects": [
                                    {"object_type": "sofa"},
                                    {"object_type": "floor_lamp"},
                                ],
                            }
                        ],
                    },
                    {
                        "cluster_id": "lighting_support",
                        "required_bundles": [
                            {
                                "bundle_id": "lighting",
                                "objects": [{"object_type": "floor_lamp"}],
                            }
                        ],
                    },
                ]
            },
            "notes": [],
        }

        cleaned = orchestrator._validate_semantic_cluster_unique_members(payload)

        self.assertEqual(len(cleaned["clusters"]), 1)
        self.assertEqual(cleaned["clusters"][0]["cluster_id"], "main_seating")
        active = cleaned["semantic_layout_program"]["active_clusters"]
        self.assertEqual([row["cluster_id"] for row in active], ["main_seating"])
        self.assertTrue(
            any(
                "Removed duplicate member floor_lamp" in note
                for note in cleaned["notes"]
            )
        )

    def test_log_seed_relation_output_warns_on_duplicate_members(self) -> None:
        cluster_output = {
            "clusters": [
                {
                    "cluster_id": "sleep_zone",
                    "tag": "sleep",
                    "members": ["bed", "chair"],
                    "anchors": ["bed"],
                },
                {
                    "cluster_id": "work_zone",
                    "tag": "work",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                },
            ]
        }

        with self.assertLogs(orchestrator.logger, level="INFO") as captured:
            orchestrator._log_seed_relation_output(
                cluster_output,
                context="primary",
            )

        joined = "\n".join(captured.output)
        self.assertIn("Seed relation planner primary produced 2 cluster(s).", joined)
        self.assertIn("duplicate members across clusters", joined)
        self.assertIn("chair", joined)

    def test_concept_repair_raw_zero_builds_zone_spread_without_topology_change(
        self,
    ) -> None:
        relation_plan = _repair_relation_plan()
        original_relation_plan = deepcopy(relation_plan)

        attempts = orchestrator._build_concept_repair_attempts(
            concept={"concept_id": "concept_01"},
            relation_plan=relation_plan,
            merged_output=_repair_merged_output(),
            solver_output=_unsat_solver_output(
                reasons=["layout_incomplete"],
                offenders=["work_study"],
                raw_solution_count=0,
            ),
        )

        self.assertGreaterEqual(len(attempts), 1)
        self.assertEqual(attempts[0].repair_id, "zone_spread")
        self.assertEqual(relation_plan, original_relation_plan)
        self.assertEqual(
            attempts[0].relation_plan["macro_constraints"]["protected_topology"],
            original_relation_plan["macro_constraints"]["protected_topology"],
        )
        repaired_work = attempts[0].relation_plan[
            "anchor_region_preferences_by_cluster"
        ]["work_study"]
        self.assertNotEqual(repaired_work["zone_assignment"], "right_wall_zone")
        self.assertEqual(repaired_work["preferred_wall_side"], "")

    def test_zero_anchor_repair_softens_only_primary_corridor_for_offender(
        self,
    ) -> None:
        relation_plan = _repair_relation_plan()
        relation_plan["macro_constraints"]["protected_topology"] = [
            {
                "region": "door_1_entry_clearance",
                "zone_type": "entry_landing",
                "enforcement": "hard",
            },
            {
                "region": "door_1_to_room_center_corridor",
                "zone_type": "primary_circulation_corridor",
                "enforcement": "hard_soft",
            },
        ]
        merged_output = _repair_merged_output()
        merged_output["clusters"][0]["cluster_rules"]["zone_claims"] = {
            "avoid_regions": ["door_1_entry_clearance", "entry_to_center_corridor"]
        }
        original_relation_plan = deepcopy(relation_plan)

        attempts = orchestrator._build_concept_repair_attempts(
            concept={"concept_id": "concept_01"},
            relation_plan=relation_plan,
            merged_output=merged_output,
            solver_output=_unsat_solver_output(
                reasons=["layout_incomplete"],
                offenders=["work_study"],
                raw_solution_count=0,
                anchor_candidate_counts={"sleep_core": 8, "work_study": 0},
            ),
        )

        self.assertGreaterEqual(len(attempts), 1)
        self.assertEqual(attempts[0].repair_id, "primary_corridor_softening")
        self.assertEqual(relation_plan, original_relation_plan)
        repaired_plan = attempts[0].relation_plan
        repaired_work = repaired_plan["anchor_region_preferences_by_cluster"][
            "work_study"
        ]
        self.assertEqual(
            repaired_work["forbidden_region_ids"],
            ["door_1_entry_clearance"],
        )
        topology = repaired_plan["macro_constraints"]["protected_topology"]
        self.assertNotIn("softened_for_clusters", topology[0])
        self.assertEqual(topology[1]["softened_for_clusters"], ["work_study"])
        repaired_claims = attempts[0].merged_output["clusters"][0]["cluster_rules"][
            "zone_claims"
        ]
        self.assertEqual(
            repaired_claims["avoid_regions"],
            ["door_1_entry_clearance"],
        )

    def test_anchor_pair_spread_repair_relaxes_less_primary_pair_member(
        self,
    ) -> None:
        relation_plan = _repair_relation_plan()
        prefs = relation_plan["anchor_region_preferences_by_cluster"]
        storage_pref = deepcopy(prefs.pop("work_study"))
        storage_pref.update(
            {
                "role_kind": "storage",
                "scenario_role": "secondary",
                "zone_assignment": "right_wall_zone",
                "committed_region_ids": ["right_wall_zone"],
                "preferred_region_ids": [
                    "right_wall_zone",
                    "edge_loading_zone",
                    "bottom_wall_zone",
                ],
            }
        )
        prefs["storage_core"] = storage_pref
        relation_plan["anchor_layout_hints_by_cluster"] = deepcopy(prefs)
        relation_plan["macro_concept"]["cluster_zone_plan"] = [
            {"cluster_id": "sleep_core", "zone_assignment": "right_wall_zone"},
            {"cluster_id": "storage_core", "zone_assignment": "right_wall_zone"},
        ]
        relation_plan["object_orientations"] = [
            {
                "cluster_id": "storage_core",
                "object_id": "wardrobe",
                "target_region": "right_wall_zone",
            }
        ]
        relation_plan["macro_constraints"]["anchor_region_constraints"] = [
            {
                "cluster_id": "storage_core",
                "zone_assignment": "right_wall_zone",
                "forbidden_region_ids": [
                    "door_1_entry_clearance",
                    "entry_to_center_corridor",
                ],
            }
        ]
        relation_plan["macro_constraints"]["protected_topology"] = [
            {
                "region": "door_1_entry_clearance",
                "zone_type": "entry_landing",
                "enforcement": "hard",
            },
            {
                "region": "door_1_to_room_center_corridor",
                "zone_type": "primary_circulation_corridor",
                "enforcement": "hard_soft",
            },
        ]
        original_relation_plan = deepcopy(relation_plan)

        attempts = orchestrator._build_concept_repair_attempts(
            concept=deepcopy(relation_plan["macro_concept"]),
            relation_plan=relation_plan,
            merged_output=_repair_merged_output(),
            solver_output=_unsat_solver_output(
                reasons=["layout_incomplete"],
                offenders=["sleep_core", "storage_core"],
                raw_solution_count=0,
                anchor_candidate_counts={"sleep_core": 5, "storage_core": 11},
                failure_kind="no_compatible_anchor_set",
                pairwise_incompatible_anchor_pairs=[
                    {
                        "cluster_a": "sleep_core",
                        "cluster_b": "storage_core",
                        "candidate_pairs": 55,
                        "non_overlap_pairs": 0,
                        "overlap_pairs": 55,
                    }
                ],
            ),
        )

        self.assertGreaterEqual(len(attempts), 1)
        self.assertEqual(attempts[0].repair_id, "anchor_pair_spread")
        self.assertEqual(relation_plan, original_relation_plan)
        repaired_plan = attempts[0].relation_plan
        repaired_storage = repaired_plan["anchor_region_preferences_by_cluster"][
            "storage_core"
        ]
        self.assertEqual(repaired_storage["zone_assignment"], "edge_loading_zone")
        self.assertEqual(
            repaired_storage["committed_region_ids"], ["edge_loading_zone"]
        )
        self.assertEqual(repaired_storage["preferred_wall_side"], "")
        self.assertEqual(repaired_storage["committed_wall_side"], "")
        self.assertEqual(repaired_storage["anchor_strength"], "medium")
        self.assertEqual(repaired_storage["placement_bias"], "balanced")
        self.assertEqual(
            repaired_storage["placement_behavior"]["wall_backing"],
            "preferred",
        )
        self.assertEqual(
            repaired_storage["placement_behavior"]["pose_flexibility"],
            "medium",
        )
        self.assertEqual(
            repaired_plan["anchor_layout_hints_by_cluster"]["storage_core"][
                "zone_assignment"
            ],
            "edge_loading_zone",
        )
        self.assertEqual(
            repaired_plan["object_orientations"][0]["target_region"],
            "edge_loading_zone",
        )
        constraints = repaired_plan["macro_constraints"]["anchor_region_constraints"]
        self.assertEqual(
            constraints[0]["forbidden_region_ids"],
            ["door_1_entry_clearance"],
        )
        topology = repaired_plan["macro_constraints"]["protected_topology"]
        self.assertNotIn("softened_for_clusters", topology[0])
        self.assertEqual(topology[1]["softened_for_clusters"], ["storage_core"])
        self.assertEqual(
            topology[1]["relaxation_reason"],
            "anchor_pair_spread_corridor_softening_after_strict_solver_failure",
        )

    def test_concept_repair_functional_gate_relaxes_only_offending_cluster(
        self,
    ) -> None:
        relation_plan = _repair_relation_plan()
        relation_plan["anchor_region_preferences_by_cluster"]["sleep_core"][
            "zone_assignment"
        ] = "quiet_private_deep_zone"
        relation_plan["anchor_region_preferences_by_cluster"]["sleep_core"][
            "preferred_wall_side"
        ] = ""

        attempts = orchestrator._build_concept_repair_attempts(
            concept={"concept_id": "concept_01"},
            relation_plan=relation_plan,
            merged_output={"clusters": []},
            solver_output=_unsat_solver_output(
                reasons=["view_corridor_blocked"],
                offenders=["work_study"],
                raw_solution_count=192,
            ),
        )

        functional = next(
            attempt for attempt in attempts if attempt.repair_id == "functional_gate"
        )
        repaired_prefs = functional.relation_plan[
            "anchor_region_preferences_by_cluster"
        ]
        self.assertEqual(
            repaired_prefs["sleep_core"]["zone_assignment"],
            "quiet_private_deep_zone",
        )
        self.assertEqual(
            repaired_prefs["work_study"]["zone_assignment"],
            "edge_loading_zone",
        )
        self.assertEqual(
            functional.relation_plan["object_orientations"][0]["target_region"],
            "edge_loading_zone",
        )
        self.assertIn(
            "door_1_entry_clearance",
            repaired_prefs["work_study"]["forbidden_region_ids"],
        )

    def test_support_droppable_repair_keeps_dominant_anchor_protected(self) -> None:
        merged_output = _repair_merged_output()
        relation_plan = _repair_relation_plan()
        concept: dict[str, object] = {"object_solver_policy": {}}

        changed = orchestrator._apply_support_droppable_repair(
            merged_output=merged_output,
            relation_plan=relation_plan,
            concept=concept,
        )

        self.assertTrue(changed)
        policy = merged_output["clusters"][0]["cluster_rules"]["anchor_first_policy"]
        self.assertEqual(policy["protected_ids"], ["desk"])
        self.assertEqual(policy["droppable_ids"], ["chair"])
        relation_policy = relation_plan["object_solver_policy"]
        self.assertEqual(
            relation_policy["protected_ids_by_cluster"]["work_study"],
            ["desk"],
        )
        self.assertEqual(
            relation_policy["droppable_ids_by_cluster"]["work_study"],
            ["chair"],
        )

    def test_support_droppable_repair_keeps_hard_requested_support_protected(
        self,
    ) -> None:
        merged_output = _repair_merged_output()
        cluster = merged_output["clusters"][0]
        cluster["decisions"] = [
            {"object_type": "desk", "min_keep": 1},
            {
                "object_type": "chair",
                "min_keep": 1,
                "request_contract_intent": "must_keep",
            },
        ]
        cluster["object_program"] = {
            "required_object_ids": ["desk", "chair"],
            "protected_ids": ["desk", "chair"],
            "droppable_ids": [],
        }
        relation_plan = _repair_relation_plan()
        concept: dict[str, object] = {"object_solver_policy": {}}

        changed = orchestrator._apply_support_droppable_repair(
            merged_output=merged_output,
            relation_plan=relation_plan,
            concept=concept,
        )

        self.assertFalse(changed)
        policy = cluster["cluster_rules"]["anchor_first_policy"]
        self.assertEqual(policy["protected_ids"], ["desk", "chair"])
        self.assertEqual(policy["droppable_ids"], [])
        relation_policy = relation_plan["object_solver_policy"]
        self.assertEqual(
            relation_policy["protected_ids_by_cluster"]["work_study"],
            ["desk", "chair"],
        )
        self.assertEqual(
            relation_policy["droppable_ids_by_cluster"]["work_study"],
            [],
        )

    def test_required_support_relaxation_removes_bookshelf_from_protected_only(
        self,
    ) -> None:
        merged_output = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "members": ["wardrobe", "bookshelf"],
                    "decisions": [
                        {"object_type": "wardrobe", "min_keep": 1},
                        {
                            "object_type": "bookshelf",
                            "min_keep": 1,
                            "request_contract_intent": "must_keep",
                        },
                    ],
                    "cluster_rules": {
                        "anchor_first_policy": {
                            "protected_ids": ["wardrobe", "bookshelf"],
                            "droppable_ids": ["bookshelf"],
                        }
                    },
                    "object_program": {
                        "members": ["wardrobe", "bookshelf"],
                        "required_object_ids": ["wardrobe", "bookshelf"],
                        "required_support_ids": ["bookshelf"],
                        "protected_ids": ["wardrobe", "bookshelf"],
                        "droppable_ids": ["bookshelf"],
                        "object_specs_by_id": {"bookshelf": {"required_support": True}},
                    },
                }
            ]
        }
        relation_plan = {
            "object_solver_policy": {
                "protected_ids_by_cluster": {"storage_core": ["wardrobe", "bookshelf"]},
                "droppable_ids_by_cluster": {"storage_core": ["bookshelf"]},
            }
        }
        concept: dict[str, object] = {"object_solver_policy": {}}

        changed = orchestrator._apply_required_support_relaxation(
            merged_output=merged_output,
            relation_plan=relation_plan,
            concept=concept,
        )

        self.assertTrue(changed)
        object_program = merged_output["clusters"][0]["object_program"]
        self.assertEqual(object_program["protected_ids"], ["wardrobe"])
        self.assertEqual(object_program["droppable_ids"], [])
        self.assertEqual(object_program["required_support_ids"], ["bookshelf"])
        policy = relation_plan["object_solver_policy"]
        self.assertEqual(
            policy["protected_ids_by_cluster"]["storage_core"],
            ["wardrobe"],
        )
        self.assertEqual(
            policy["required_support_ids_by_cluster"]["storage_core"],
            ["bookshelf"],
        )

    def test_storage_front_token_repair_preserves_valid_storage_front(self) -> None:
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "object_specs_by_id": {"wardrobe": {"front": "top"}},
                    },
                    "cluster_rules": {
                        "facing": {"wardrobe": {"front": "top"}},
                    },
                },
                {
                    "cluster_id": "sleep_core",
                    "anchors": ["bed"],
                    "object_program": {
                        "dominant_anchor_id": "bed",
                        "object_specs_by_id": {"bed": {"front": "top"}},
                    },
                    "cluster_rules": {
                        "facing": {"bed": {"front": "top"}},
                    },
                },
            ]
        }

        changed = orchestrator._apply_storage_front_token_repair(
            merged_output=merged_output,
            diagnostics={"offending_clusters": ["storage_core"]},
        )

        self.assertFalse(changed)
        clusters = merged_output["clusters"]
        storage = clusters[0]
        sleep = clusters[1]
        self.assertEqual(
            storage["object_program"]["object_specs_by_id"]["wardrobe"]["front"],
            "top",
        )
        self.assertEqual(
            storage["cluster_rules"]["facing"]["wardrobe"]["front"],
            "top",
        )
        self.assertEqual(
            sleep["object_program"]["object_specs_by_id"]["bed"]["front"],
            "top",
        )

    def test_storage_front_token_repair_normalizes_missing_storage_front(self) -> None:
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "object_specs_by_id": {"wardrobe": {"front": "sideways"}},
                    },
                    "cluster_rules": {
                        "facing": {"wardrobe": {}},
                    },
                },
                {
                    "cluster_id": "sleep_core",
                    "anchors": ["bed"],
                    "object_program": {
                        "dominant_anchor_id": "bed",
                        "object_specs_by_id": {"bed": {"front": "sideways"}},
                    },
                    "cluster_rules": {
                        "facing": {"bed": {"front": "sideways"}},
                    },
                },
            ]
        }

        changed = orchestrator._apply_storage_front_token_repair(
            merged_output=merged_output,
            diagnostics={"offending_clusters": ["storage_core", "sleep_core"]},
        )

        self.assertTrue(changed)
        clusters = merged_output["clusters"]
        storage = clusters[0]
        sleep = clusters[1]
        self.assertEqual(
            storage["object_program"]["object_specs_by_id"]["wardrobe"]["front"],
            "top",
        )
        self.assertEqual(
            storage["cluster_rules"]["facing"]["wardrobe"]["front"],
            "top",
        )
        self.assertEqual(
            sleep["object_program"]["object_specs_by_id"]["bed"]["front"],
            "sideways",
        )

    def test_anchor_pair_spread_does_not_rewrite_valid_storage_front(self) -> None:
        relation_plan: dict[str, object] = {
            "anchor_region_preferences_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "left_wall_zone",
                    "preferred_region_ids": ["left_wall_zone"],
                    "anchor_strength": "strong",
                    "role_kind": "sleep",
                },
                "storage_core": {
                    "zone_assignment": "right_wall_zone",
                    "preferred_region_ids": [
                        "right_wall_zone",
                        "edge_loading_zone",
                    ],
                    "preferred_wall_side": "right_wall",
                    "anchor_strength": "strong",
                    "placement_bias": "wall_backed",
                    "placement_behavior": {
                        "wall_backing": "required",
                        "pose_flexibility": "low",
                    },
                    "role_kind": "storage",
                },
            },
            "macro_concept": {
                "cluster_zone_plan": [
                    {"cluster_id": "sleep_core", "zone_assignment": "right_wall_zone"},
                    {
                        "cluster_id": "storage_core",
                        "zone_assignment": "right_wall_zone",
                    },
                ],
            },
            "macro_constraints": {},
        }
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "object_specs_by_id": {"wardrobe": {"front": "top"}},
                    },
                    "cluster_rules": {
                        "facing": {"wardrobe": {"front": "top"}},
                    },
                }
            ]
        }

        attempt = orchestrator._make_concept_repair_attempt(
            repair_id="anchor_pair_spread",
            concept=deepcopy(relation_plan["macro_concept"]),
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics={
                "offending_clusters": ["storage_core"],
                "pairwise_incompatible_anchor_pairs": [
                    {
                        "cluster_a": "sleep_core",
                        "cluster_b": "storage_core",
                        "candidate_pairs": 12,
                        "non_overlap_pairs": 0,
                    }
                ],
                "quality_gate_reasons": ["layout_incomplete"],
            },
            apply_anchor_pair_spread=True,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            zone_pileup_clusters=set(),
        )

        self.assertIsNotNone(attempt)
        if attempt is None:
            raise AssertionError("anchor_pair_spread should produce a repair attempt")
        storage = attempt.merged_output["clusters"][0]
        self.assertEqual(
            storage["object_program"]["object_specs_by_id"]["wardrobe"]["front"],
            "top",
        )
        self.assertEqual(
            storage["cluster_rules"]["facing"]["wardrobe"]["front"],
            "top",
        )
        self.assertNotIn(
            "Normalized missing storage-anchor front metadata.",
            attempt.notes,
        )

    def test_secondary_front_access_relaxation_softens_storage_not_bed(self) -> None:
        relation_plan: dict[str, object] = {
            "anchor_region_preferences_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "right_wall_zone",
                    "preferred_region_ids": ["right_wall_zone"],
                    "role_kind": "sleep",
                    "scenario_role": "primary",
                },
                "storage_core": {
                    "zone_assignment": "edge_loading_zone",
                    "preferred_region_ids": ["edge_loading_zone"],
                    "role_kind": "storage",
                    "scenario_role": "support",
                },
            },
            "macro_concept": {"cluster_zone_plan": []},
        }
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "sleep_core",
                    "anchors": ["bed"],
                    "object_program": {
                        "dominant_anchor_id": "bed",
                        "required_object_ids": ["bed"],
                        "protected_ids": ["bed"],
                    },
                    "cluster_rules": {
                        "access_requirements": [
                            {"id": "bed", "type": "front_clearance", "required": True}
                        ]
                    },
                },
                {
                    "cluster_id": "storage_core",
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "required_object_ids": ["wardrobe"],
                        "protected_ids": ["wardrobe"],
                    },
                    "cluster_rules": {
                        "access_requirements": [
                            {
                                "id": "wardrobe",
                                "type": "front_clearance",
                                "required": True,
                            }
                        ]
                    },
                },
            ]
        }
        original_merged = deepcopy(merged_output)

        attempt = orchestrator._make_concept_repair_attempt(
            repair_id="secondary_front_access_relaxation",
            concept=deepcopy(relation_plan["macro_concept"]),
            relation_plan=relation_plan,
            merged_output=merged_output,
            diagnostics={
                "quality_gate_reasons": ["front_access_zone_blocked"],
                "front_access_blocking_pairs": [
                    {
                        "blocker_cluster_id": "sleep_core",
                        "blocker_object_id": "bed",
                        "blocked_cluster_id": "storage_core",
                        "blocked_object_id": "wardrobe",
                    }
                ],
            },
            apply_anchor_pair_spread=False,
            apply_primary_corridor_relaxation=False,
            apply_zone_spread=False,
            apply_functional_gate=False,
            apply_required_support_relaxation=False,
            apply_support_droppable=False,
            apply_secondary_front_access_relaxation=True,
            zone_pileup_clusters=set(),
        )

        self.assertIsNotNone(attempt)
        if attempt is None:
            raise AssertionError("secondary front access repair should be built")
        self.assertEqual(merged_output, original_merged)
        repaired_clusters = {
            cluster["cluster_id"]: cluster
            for cluster in attempt.merged_output["clusters"]
        }
        bed_access = repaired_clusters["sleep_core"]["cluster_rules"][
            "access_requirements"
        ][0]
        wardrobe_access = repaired_clusters["storage_core"]["cluster_rules"][
            "access_requirements"
        ][0]
        self.assertTrue(bed_access["required"])
        self.assertFalse(wardrobe_access["required"])
        self.assertEqual(
            wardrobe_access["relaxation_reason"],
            "front_access_pair_conflict_after_anchor_pair_spread",
        )
        self.assertEqual(
            repaired_clusters["storage_core"]["object_program"]["required_object_ids"],
            ["wardrobe"],
        )
        self.assertEqual(
            repaired_clusters["storage_core"]["object_program"]["protected_ids"],
            ["wardrobe"],
        )

    def test_secondary_storage_wall_rehome_runs_before_front_relaxation(self) -> None:
        relation_plan: dict[str, object] = {
            "anchor_region_preferences_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "left_wall_zone",
                    "preferred_region_ids": ["left_wall_zone"],
                    "role_kind": "sleep",
                    "scenario_role": "primary",
                },
                "storage_core": {
                    "zone_assignment": "right_wall_zone",
                    "committed_region_ids": ["right_wall_zone"],
                    "preferred_region_ids": [
                        "right_wall_zone",
                        "top_wall_zone",
                    ],
                    "preferred_wall_side": "right_wall",
                    "committed_wall_side": "right_wall",
                    "anchor_strength": "strong",
                    "placement_bias": "wall_backed",
                    "placement_behavior": {
                        "wall_backing": "required",
                        "front_space": "required",
                        "pose_flexibility": "low",
                    },
                    "role_kind": "storage",
                    "scenario_role": "secondary",
                },
            },
            "macro_concept": {
                "cluster_zone_plan": [
                    {
                        "cluster_id": "storage_core",
                        "zone_assignment": "right_wall_zone",
                    }
                ],
            },
            "macro_constraints": {
                "anchor_region_constraints": [
                    {
                        "cluster_id": "storage_core",
                        "zone_assignment": "right_wall_zone",
                    }
                ]
            },
        }
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "members": ["wardrobe"],
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "required_object_ids": ["wardrobe"],
                        "protected_ids": ["wardrobe"],
                    },
                    "cluster_rules": {
                        "access_requirements": [
                            {
                                "id": "wardrobe",
                                "type": "front_clearance",
                                "required": True,
                            }
                        ]
                    },
                }
            ]
        }
        original_relation_plan = deepcopy(relation_plan)
        attempts = orchestrator._build_concept_repair_attempts(
            concept=deepcopy(relation_plan["macro_concept"]),
            relation_plan=relation_plan,
            merged_output=merged_output,
            solver_output=_front_access_unsat_with_storage_wall_option(),
        )

        self.assertTrue(attempts)
        self.assertEqual(attempts[0].repair_id, "secondary_storage_wall_rehome")
        repaired_plan = attempts[0].relation_plan
        storage_pref = repaired_plan["anchor_region_preferences_by_cluster"][
            "storage_core"
        ]
        self.assertEqual(storage_pref["zone_assignment"], "top_wall_zone")
        self.assertEqual(storage_pref["committed_region_ids"], ["top_wall_zone"])
        self.assertEqual(storage_pref["committed_wall_side"], "top_wall")
        self.assertEqual(storage_pref["preferred_wall_side"], "top_wall")
        self.assertEqual(storage_pref["anchor_strength"], "medium")
        self.assertEqual(
            storage_pref["placement_behavior"]["wall_backing"],
            "preferred",
        )
        self.assertEqual(
            storage_pref["placement_behavior"]["pose_flexibility"],
            "medium",
        )
        storage = attempts[0].merged_output["clusters"][0]
        self.assertTrue(storage["cluster_rules"]["access_requirements"][0]["required"])
        self.assertEqual(relation_plan, original_relation_plan)

    def test_repair_queue_chains_anchor_pair_to_secondary_front_access(self) -> None:
        relation_plan: dict[str, object] = {
            "anchor_region_preferences_by_cluster": {
                "sleep_core": {
                    "zone_assignment": "left_wall_zone",
                    "preferred_region_ids": ["left_wall_zone"],
                    "anchor_strength": "strong",
                    "role_kind": "sleep",
                    "scenario_role": "primary",
                },
                "storage_core": {
                    "zone_assignment": "right_wall_zone",
                    "preferred_region_ids": [
                        "right_wall_zone",
                        "edge_loading_zone",
                    ],
                    "preferred_wall_side": "right_wall",
                    "anchor_strength": "strong",
                    "placement_bias": "wall_backed",
                    "placement_behavior": {
                        "wall_backing": "required",
                        "pose_flexibility": "low",
                    },
                    "role_kind": "storage",
                    "scenario_role": "support",
                },
            },
            "macro_concept": {
                "cluster_zone_plan": [
                    {"cluster_id": "sleep_core", "zone_assignment": "left_wall_zone"},
                    {
                        "cluster_id": "storage_core",
                        "zone_assignment": "right_wall_zone",
                    },
                ],
            },
            "macro_constraints": {},
        }
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "members": ["wardrobe"],
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "required_object_ids": ["wardrobe"],
                        "protected_ids": ["wardrobe"],
                    },
                    "cluster_rules": {
                        "access_requirements": [
                            {
                                "id": "wardrobe",
                                "type": "front_clearance",
                                "required": True,
                            }
                        ]
                    },
                }
            ]
        }
        solver = _FakeObjectLevelSolver(
            [
                _unsat_solver_output(
                    reasons=["layout_incomplete"],
                    offenders=["sleep_core", "storage_core"],
                    raw_solution_count=0,
                    anchor_candidate_counts={"sleep_core": 5, "storage_core": 8},
                    failure_kind="no_compatible_anchor_set",
                    pairwise_incompatible_anchor_pairs=[
                        {
                            "cluster_a": "sleep_core",
                            "cluster_b": "storage_core",
                            "candidate_pairs": 40,
                            "non_overlap_pairs": 0,
                            "overlap_pairs": 40,
                        }
                    ],
                ),
                _front_access_unsat_solver_output(),
                _ok_object_solver_output(),
            ]
        )

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept=deepcopy(relation_plan["macro_concept"]),
            relation_plan=relation_plan,
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=merged_output,
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="full",
        )

        self.assertEqual(len(solve_result.bundles), 1)
        self.assertEqual(len(solver.calls), 3)
        bundle = solve_result.bundles[0]
        self.assertEqual(
            bundle["concept_repair_attempt"],
            "secondary_front_access_relaxation",
        )
        repaired_merged = solver.calls[2]["merged_clusters_json"]
        storage = repaired_merged["clusters"][0]
        self.assertFalse(storage["cluster_rules"]["access_requirements"][0]["required"])

    def test_secondary_daylight_quality_relaxation_marks_storage_object(self) -> None:
        relation_plan: dict[str, object] = {
            "anchor_region_preferences_by_cluster": {
                "storage_core": {
                    "zone_assignment": "right_wall_zone",
                    "preferred_region_ids": ["right_wall_zone", "bottom_wall_zone"],
                    "role_kind": "storage",
                    "scenario_role": "support",
                }
            },
            "macro_concept": {
                "cluster_zone_plan": [
                    {
                        "cluster_id": "storage_core",
                        "zone_assignment": "right_wall_zone",
                    }
                ],
            },
            "macro_constraints": {},
        }
        merged_output: dict[str, object] = {
            "clusters": [
                {
                    "cluster_id": "storage_core",
                    "members": ["wardrobe"],
                    "anchors": ["wardrobe"],
                    "object_program": {
                        "dominant_anchor_id": "wardrobe",
                        "required_object_ids": ["wardrobe"],
                        "protected_ids": ["wardrobe"],
                        "object_specs_by_id": {
                            "wardrobe": {
                                "object_id": "wardrobe",
                                "category": "wardrobe",
                                "protected": True,
                                "droppable": False,
                            }
                        },
                    },
                    "cluster_rules": {},
                }
            ]
        }
        solver = _FakeObjectLevelSolver(
            [_daylight_quality_unsat_solver_output(), _ok_object_solver_output()]
        )

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept={"concept_id": "concept_01", "concept_family": "edge_weighted"},
            relation_plan=relation_plan,
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=merged_output,
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="full",
        )

        self.assertEqual(len(solve_result.bundles), 1)
        self.assertEqual(len(solver.calls), 2)
        repaired = solver.calls[1]["merged_clusters_json"]["clusters"][0]
        object_program = repaired["object_program"]
        self.assertEqual(object_program["protected_ids"], ["wardrobe"])
        self.assertEqual(object_program["required_object_ids"], ["wardrobe"])
        wardrobe = object_program["object_specs_by_id"]["wardrobe"]
        self.assertTrue(wardrobe["daylight_quality_relaxed"])

    def test_object_level_variant_retry_uses_repair_when_initial_unsat(self) -> None:
        solver = _FakeObjectLevelSolver(
            [
                _unsat_solver_output(
                    reasons=["view_corridor_blocked"],
                    offenders=["work_study"],
                    raw_solution_count=0,
                ),
                _ok_object_solver_output(),
            ]
        )

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept={"concept_id": "concept_01", "concept_family": "daylight"},
            relation_plan=_repair_relation_plan(),
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=_repair_merged_output(),
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="full",
        )
        bundles = solve_result.bundles

        self.assertEqual(len(bundles), 1)
        self.assertTrue(solve_result.failure_diagnostics)
        self.assertEqual(len(solver.calls), 2)
        self.assertTrue(bundles[0]["concept_repair_applied"])
        self.assertEqual(bundles[0]["concept_repair_attempt"], "zone_spread")
        self.assertTrue(bundles[0]["solver_output"]["concept_repair_applied"])
        repaired_relation = solver.calls[1]["relation_plan_json"]
        self.assertNotEqual(
            repaired_relation["anchor_region_preferences_by_cluster"]["work_study"][
                "zone_assignment"
            ],
            "right_wall_zone",
        )

    def test_object_level_variant_invokes_anchor_pair_spread_repair(
        self,
    ) -> None:
        solver = _FakeObjectLevelSolver(
            [
                _unsat_solver_output(
                    reasons=["layout_incomplete"],
                    offenders=["sleep_core", "work_study"],
                    raw_solution_count=0,
                    anchor_candidate_counts={"sleep_core": 5, "work_study": 8},
                    failure_kind="no_compatible_anchor_set",
                    pairwise_incompatible_anchor_pairs=[
                        {
                            "cluster_a": "sleep_core",
                            "cluster_b": "work_study",
                            "candidate_pairs": 40,
                            "non_overlap_pairs": 0,
                            "overlap_pairs": 40,
                        }
                    ],
                ),
                _ok_object_solver_output(),
            ]
        )

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept={"concept_id": "concept_01", "concept_family": "daylight"},
            relation_plan=_repair_relation_plan(),
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=_repair_merged_output(),
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="full",
        )
        bundles = solve_result.bundles

        self.assertEqual(len(bundles), 1)
        self.assertEqual(len(solver.calls), 2)
        self.assertTrue(bundles[0]["concept_repair_applied"])
        self.assertEqual(bundles[0]["concept_repair_attempt"], "anchor_pair_spread")
        repaired_relation = solver.calls[1]["relation_plan_json"]
        repaired_work = repaired_relation["anchor_region_preferences_by_cluster"][
            "work_study"
        ]
        self.assertEqual(repaired_work["zone_assignment"], "edge_loading_zone")
        self.assertEqual(repaired_work["committed_region_ids"], ["edge_loading_zone"])

    def test_object_level_variant_ok_does_not_call_repair(self) -> None:
        solver = _FakeObjectLevelSolver([_ok_object_solver_output()])

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept={"concept_id": "concept_01", "concept_family": "daylight"},
            relation_plan=_repair_relation_plan(),
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=_repair_merged_output(),
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="full",
        )
        bundles = solve_result.bundles

        self.assertEqual(len(bundles), 1)
        self.assertEqual(solve_result.failure_diagnostics, [])
        self.assertEqual(len(solver.calls), 1)
        self.assertNotIn("concept_repair_applied", bundles[0])

    def test_object_level_variant_returns_empty_when_all_repairs_fail(self) -> None:
        solver = _FakeObjectLevelSolver(
            [
                _unsat_solver_output(
                    reasons=["view_corridor_blocked"],
                    offenders=["work_study"],
                    raw_solution_count=0,
                ),
                _unsat_solver_output(
                    reasons=["view_corridor_blocked"],
                    offenders=["work_study"],
                    raw_solution_count=192,
                ),
                _unsat_solver_output(
                    reasons=["view_corridor_blocked"],
                    offenders=["work_study"],
                    raw_solution_count=192,
                ),
                _unsat_solver_output(
                    reasons=["view_corridor_blocked"],
                    offenders=["work_study"],
                    raw_solution_count=192,
                ),
            ]
        )

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept={"concept_id": "concept_01", "concept_family": "daylight"},
            relation_plan=_repair_relation_plan(),
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=_repair_merged_output(),
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="full",
        )

        self.assertEqual(solve_result.bundles, [])
        self.assertTrue(solve_result.failure_diagnostics)
        self.assertLessEqual(
            len(solver.calls),
            orchestrator._CONCEPT_REPAIR_MAX_ATTEMPTS + 1,
        )

    def test_no_concept_repair_ablation_skips_repair_attempts(self) -> None:
        solver = _FakeObjectLevelSolver(
            [
                _unsat_solver_output(
                    reasons=["view_corridor_blocked"],
                    offenders=["work_study"],
                    raw_solution_count=0,
                ),
                _ok_object_solver_output(),
            ]
        )

        solve_result = orchestrator._solve_object_level_variant_bundle(
            concept={"concept_id": "concept_01", "concept_family": "daylight"},
            relation_plan=_repair_relation_plan(),
            room_output={"room": {"room_id": "room_1"}},
            cluster_output={"clusters": []},
            tier_output={"decisions": []},
            merged_output=_repair_merged_output(),
            solver=solver,
            manual_placements=None,
            variant_index=1,
            ablation_mode="no_concept_repair",
        )

        self.assertEqual(solve_result.bundles, [])
        self.assertTrue(solve_result.failure_diagnostics)
        self.assertEqual(len(solver.calls), 1)

    def test_no_stylist_llm_ablation_applies_empty_style_plan(self) -> None:
        class _FakeStylist:
            def __init__(self) -> None:
                self.generate_calls = 0
                self.applied_style_plan: dict[str, object] | None = None

            def generate_style_plan(self, **_kwargs: object) -> dict[str, object]:
                self.generate_calls += 1
                raise AssertionError("style LLM should be skipped")

            def apply_style_plan(self, **kwargs: object) -> dict[str, object]:
                style_plan = kwargs.get("style_plan")
                self.applied_style_plan = (
                    dict(style_plan) if isinstance(style_plan, dict) else None
                )
                return {"room": {}, "objects": []}

        stylist = _FakeStylist()

        result = orchestrator._style_judged_variant_bundle(
            bundle={
                "absolute_layout": {"status": "OK", "objects": []},
                "source": "concept:1",
            },
            stylist=stylist,
            stylist_user_context_json={},
            stylist_tenant_id=None,
            manual_placements=None,
            variant_index=1,
            ablation_mode="no_stylist_llm",
        )

        self.assertIsNotNone(result)
        self.assertEqual(stylist.generate_calls, 0)
        self.assertEqual(stylist.applied_style_plan, {})
        assert result is not None
        self.assertEqual(result["stylist_model_name"], "deterministic_fallback")

    def test_no_accessory_refill_ablation_returns_bundle_unchanged(self) -> None:
        bundle = {
            "absolute_layout": {"status": "OK", "objects": []},
            "cluster_output": {"clusters": []},
            "dropped_inventory_by_cluster": {"cluster": []},
        }

        result = orchestrator._refill_judged_variant_accessories(
            bundle=bundle,
            room_output={"room": {"room_id": "room_1"}},
            variant_index=1,
            ablation_mode="no_accessory_refill",
        )

        self.assertIs(result, bundle)

    def test_no_gallery_selector_ablation_keeps_solver_order(self) -> None:
        def _raise_selector(**_kwargs: object) -> list[dict[str, object]]:
            raise AssertionError("gallery selector should be skipped")

        original_selector = orchestrator.select_distinct_final_gallery_candidates
        orchestrator.select_distinct_final_gallery_candidates = _raise_selector
        try:
            selected = orchestrator._select_final_styling_candidates(
                [{"id": "first"}, {"id": "second"}],
                max_variants=1,
                ablation_mode="no_gallery_selector",
            )
        finally:
            orchestrator.select_distinct_final_gallery_candidates = original_selector

        self.assertEqual(selected, [{"id": "first"}])


if __name__ == "__main__":
    unittest.main()
