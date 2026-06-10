from __future__ import annotations

import importlib.util
import os
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
    "agent.cluster_composer",
    "agent.cluster_forge",
    "agent.cluster_placer",
    "agent.phase2_controller",
    "agent.room_interpreter",
    "agent.seed_concept_generator",
    "agent.solver.solver",
    "agent.stylist",
    "agent.tier_count_director",
    "agent.visual_relation_object_solver",
    "agent.visual_relation_planner",
    "cluster_composer.merge",
    "cluster_composer.outline",
    "layout.grid_policy",
    "pipeline.compose_feedback",
    "pipeline.dropped_inventory",
    "pipeline.layout_eligibility",
    "pipeline.layout_variants",
    "pipeline.manual_placements",
    "pipeline.seed_variant_policy",
]


_stub_module(
    "agent.cluster_composer",
    ClusterComposer=object,
    _build_cluster_output_from_placements=lambda **kwargs: kwargs,
)
_stub_module("agent.cluster_forge", ClusterForge=object)
_stub_module("agent.cluster_placer", build_phase2_payload=lambda **kwargs: kwargs)
_stub_module(
    "agent.phase2_controller",
    Phase2Controller=object,
    build_phase2_preview_candidate=lambda **kwargs: {},
)
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
_stub_module("agent.visual_relation_object_solver", VisualRelationObjectSolver=object)
_stub_module(
    "agent.visual_relation_planner",
    VisualRelationBundlePlanner=object,
    VisualRelationPlanner=object,
)
_stub_module(
    "cluster_composer.merge",
    merge_cluster_outputs=lambda cluster_output, tier_output: {
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
_stub_module(
    "cluster_composer.outline",
    compute_cluster_outline=lambda cluster_result: cluster_result,
)
_stub_module("layout.grid_policy", GLOBAL_LAYOUT_GRID_MM=50)
_stub_module(
    "pipeline.compose_feedback",
    apply_compose_backoff=lambda **kwargs: (kwargs.get("tier_output") or {}, [], False),
    apply_variant_diversification=lambda **kwargs: (
        kwargs.get("tier_output") or {},
        [],
        False,
    ),
    evaluate_composed_cluster_feasibility=lambda **kwargs: {"feasible": True},
    evaluate_solver_cluster_feasibility=lambda **kwargs: {"feasible": True},
)
_stub_module(
    "pipeline.dropped_inventory",
    collect_dropped_inventory=lambda **kwargs: {},
    collect_seed_omitted_inventory=lambda **kwargs: {},
    controlled_accessory_refill=lambda **kwargs: (
        kwargs.get("absolute_layout") or {},
        {"refill_count": 0},
    ),
    dropped_inventory_payload=lambda value: value,
    merge_dropped_inventory=lambda base, additions: {
        **(base or {}),
        **(additions or {}),
    },
    reintroduce_dropped_inventory=lambda **kwargs: (
        kwargs.get("absolute_layout") or {},
        {"inserted_count": 0},
    ),
)
_stub_module(
    "pipeline.layout_eligibility",
    annotate_layout_coverage=lambda **kwargs: kwargs.get("absolute_layout") or {},
)
_stub_module(
    "pipeline.layout_variants",
    build_final_gallery_selection_summary=lambda **kwargs: {
        "selected_count": len(kwargs.get("selected") or []),
    },
    material_macro_layout_signature=lambda _layout: "",
    select_distinct_final_gallery_candidates=lambda **kwargs: kwargs.get("candidates")
    or [],
)
_stub_module(
    "pipeline.manual_placements",
    apply_manual_placements_to_tier_output=lambda layout, *_args, **_kwargs: layout,
    append_manual_placements_guidance=lambda text, *_args, **_kwargs: text,
    augment_cluster_forge_with_manual_placements=lambda payload,
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


class _FakeBrief:
    def __init__(self, source: str, model_name: str) -> None:
        self.source = source
        self.model_name = model_name
        self.reason = f"Reason for {source}"
        self.special_notes = f"Special notes for {source}"
        self.focus_mode = "mixed"
        self.center_open_preference = "medium"
        self.support_cluster_behavior = "balanced"
        self.distribution_mode = "balanced"
        self.constraint_tokens = ("token",)


_stub_module(
    "pipeline.seed_variant_policy",
    apply_seed_variant_policy=lambda relation_plan, planner_clusters_json, brief: {
        **relation_plan,
        "seed_variant_policy": {"source": brief.source},
        "planner_seed_note": (
            (planner_clusters_json.get("seed_layout_state") or {}).get("notes") or []
        ),
    },
    build_seed_variant_briefs=lambda target_count: [
        _FakeBrief("seed_stable", "gpt-5.4-mini"),
        _FakeBrief("seed_open_center", "gpt-5.4-mini"),
    ][: max(1, int(target_count))],
)

orchestrator = _load_module(
    "pipeline.orchestrator_for_architecture_test",
    "pipeline/orchestrator.py",
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
) -> dict[str, object]:
    anchor_counts = anchor_candidate_counts or {"sleep_core": 6, "work_study": 36}
    return {
        "status": "UNSAT",
        "offending_clusters": offenders,
        "solver_debug": {
            "raw_solution_count": raw_solution_count,
            "anchor_candidate_counts": anchor_counts,
        },
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
        ) in orchestrator._REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS.items():
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

    def test_seed_retry_triggers_when_solver_gallery_diversity_collapses(self) -> None:
        collapsed_summary = {
            "returned_count": 2,
            "macro_distinct_candidate_count": 1,
            "visual_family_distinct_candidate_count": 2,
            "fallback_near_distinct_count": 0,
        }
        healthy_summary = {
            "returned_count": 5,
            "macro_distinct_candidate_count": 5,
            "visual_family_distinct_candidate_count": 5,
            "fallback_near_distinct_count": 0,
        }

        self.assertTrue(
            orchestrator._should_retry_seed_concepts(
                collapsed_summary,
                target_final_count=5,
            )
        )
        self.assertFalse(
            orchestrator._should_retry_seed_concepts(
                healthy_summary,
                target_final_count=5,
            )
        )

    def test_seed_retry_feedback_includes_projected_physical_signatures(self) -> None:
        feedback = orchestrator._seed_retry_feedback_from_selection(
            seed_relation_plans={
                "concepts": [
                    {
                        "concept_id": "concept_01",
                        "concept_family": "zoned",
                        "projected_physical_signature": "main:left|media:right",
                    }
                ]
            },
            candidates=[
                {
                    "source": "concept:concept_01",
                    "macro_layout_signature": "macro_a",
                }
            ],
            selected=[],
            selection_summary={
                "returned_count": 0,
                "candidate_count": 1,
                "macro_distinct_candidate_count": 1,
                "visual_family_distinct_candidate_count": 1,
                "fallback_near_distinct_count": 0,
                "collapsed_macro_layouts": [],
                "collapsed_visual_families": [],
                "top_quality_gate_reasons": {"view_corridor_blocked": 2},
            },
            target_final_count=5,
        )

        self.assertEqual(
            feedback["retry_reason"], "insufficient_distinct_solver_candidates"
        )
        self.assertEqual(
            feedback["used_projected_physical_signatures"],
            ["main:left|media:right"],
        )
        self.assertEqual(feedback["used_macro_layout_signatures"], ["macro_a"])

    def test_run_case_uses_deterministic_seed_and_skips_llm_retry(self) -> None:
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

        class _FakeClusterForge:
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
                uses_llm = bool(kwargs.get("use_llm", True))
                return {
                    "status": "OK",
                    "room_type": "bedroom",
                    "controller": "llm_guided_seed_concept_generator"
                    if uses_llm
                    else "deterministic_seed_concept_generator",
                    "settings": {"llm_guided": uses_llm},
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
            "ClusterForge": _FakeClusterForge,
            "TierCountDirector": _FakeTierCountDirector,
            "SeedConceptGenerator": _FakeSeedConceptGenerator,
            "Stylist": _FakeStylist,
            "_build_runtime_solver": lambda: object(),
            "prepare_input_payload_for_manual_placements": lambda payload: (
                payload,
                None,
            ),
            "merge_cluster_outputs": lambda _cluster_output, _tier_output: {
                "clusters": []
            },
            "_run_visual_relation_planner_experiment": lambda **_kwargs: {},
            "_visual_relation_v2_must_stop": lambda _experiment: False,
            "_visual_relation_candidate_bundles": lambda _experiment: [],
            "_solve_seed_concept_pool": lambda **_kwargs: [],
            "_should_retry_seed_concepts": lambda *_args, **_kwargs: True,
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
        self.assertIs(calls[0]["use_llm"], False)

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

    def test_fast_layout_mode_caps_attempt_budgets(self) -> None:
        previous = os.environ.get("TKNT_LAYOUT_SPEED_MODE")
        os.environ["TKNT_LAYOUT_SPEED_MODE"] = "fast"
        try:
            self.assertEqual(orchestrator._compose_attempt_limit(4), 2)
            self.assertEqual(orchestrator._solver_attempt_limit(8), 3)
            self.assertEqual(orchestrator._seed_diversification_attempt_limit(20), 8)
        finally:
            if previous is None:
                os.environ.pop("TKNT_LAYOUT_SPEED_MODE", None)
            else:
                os.environ["TKNT_LAYOUT_SPEED_MODE"] = previous

    def test_visual_relation_planner_v2_env_uses_macro_concept_version(self) -> None:
        previous = os.environ.get("TKNT_VISUAL_RELATION_PLANNER")
        os.environ["TKNT_VISUAL_RELATION_PLANNER"] = "v2"
        try:
            config = orchestrator._visual_relation_request_config(
                {
                    "experimental": {
                        "visual_relation_planner": {
                            "enabled": True,
                            "mode": "shadow",
                            "floorplan_image_data_url": "data:image/png;base64,abc",
                        }
                    }
                }
            )
        finally:
            if previous is None:
                os.environ.pop("TKNT_VISUAL_RELATION_PLANNER", None)
            else:
                os.environ["TKNT_VISUAL_RELATION_PLANNER"] = previous

        self.assertIsNotNone(config)
        self.assertEqual(config["planner_version"], "v2")
        self.assertEqual(
            orchestrator._visual_relation_artifact_version("v2"),
            "v2_macro_concept_bundle",
        )

    def test_visual_relation_planner_v2_skip_stops_before_deterministic(self) -> None:
        artifact = orchestrator._visual_relation_skip_artifact(
            mode="shadow",
            reason="OpenAI error",
            planner_version="v2_macro_concept_bundle",
        )

        self.assertTrue(orchestrator._visual_relation_v2_must_stop(artifact))
        self.assertIn("generation stops", artifact["notes"][0])
        self.assertFalse(
            orchestrator._visual_relation_v2_must_stop(
                {
                    "mode": "shadow",
                    "status": "SKIPPED",
                    "planner_version": "v1_object_region",
                }
            )
        )

    def test_compose_status_feedback_keeps_room_overflow_offenders(self) -> None:
        feedback = orchestrator._compose_status_only_feedback(
            {
                "feasible": False,
                "offenders": [
                    {
                        "cluster_id": "main_seating_area",
                        "reason": "CLUSTER_ENVELOPE_EXCEEDS_ROOM",
                    }
                ],
            }
        )

        self.assertFalse(feedback["feasible"])
        self.assertEqual(len(feedback["offenders"]), 1)
        self.assertEqual(
            feedback["offenders"][0]["reason"],
            "CLUSTER_ENVELOPE_EXCEEDS_ROOM",
        )

    def test_log_cluster_forge_output_warns_on_duplicate_members(self) -> None:
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
            orchestrator._log_cluster_forge_output(
                cluster_output,
                context="primary",
            )

        joined = "\n".join(captured.output)
        self.assertIn("Cluster forge primary produced 2 cluster(s).", joined)
        self.assertIn("duplicate members across clusters", joined)
        self.assertIn("chair", joined)

    def test_fast_layout_mode_no_longer_splits_clusters_before_tier_count(
        self,
    ) -> None:
        previous = os.environ.get("TKNT_LAYOUT_SPEED_MODE")
        os.environ["TKNT_LAYOUT_SPEED_MODE"] = "fast"
        try:
            cluster_output, notes = orchestrator._speed_split_cluster_output(
                room_output={
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
                },
                cluster_output={
                    "clusters": [
                        {
                            "cluster_id": "main_seating_area",
                            "tag": "living",
                            "members": [
                                "sofa",
                                "sectional_sofa",
                                "armchair",
                                "recliner",
                                "coffee_table",
                                "side_table",
                                "tv_console",
                            ],
                            "anchors": ["sofa", "tv_console"],
                            "hard_constraints": [
                                {"type": "requires_access", "id": "sofa"},
                                {"type": "requires_access", "id": "tv_console"},
                            ],
                            "soft_constraints": [
                                {
                                    "type": "prefer_facing",
                                    "a": "sofa",
                                    "b": "tv_console",
                                }
                            ],
                            "cluster_rules": {
                                "allowed_rotations": {
                                    "sofa": [0, 90],
                                    "tv_console": [0, 90],
                                },
                                "facing": {
                                    "sofa": {"front": "bottom"},
                                    "tv_console": {"front": "bottom"},
                                },
                            },
                        }
                    ]
                },
            )
        finally:
            if previous is None:
                os.environ.pop("TKNT_LAYOUT_SPEED_MODE", None)
            else:
                os.environ["TKNT_LAYOUT_SPEED_MODE"] = previous

        self.assertEqual(len(cluster_output["clusters"]), 1)
        self.assertEqual(
            cluster_output["clusters"][0]["cluster_id"], "main_seating_area"
        )
        self.assertEqual(notes, [])

    def test_solver_backoff_allows_missing_hard_valid_candidates(self) -> None:
        self.assertTrue(
            orchestrator._solver_feasibility_allows_backoff(
                {
                    "offenders": [
                        {
                            "cluster_id": "main_seating_area",
                            "reason": "SOLVER_HAS_NO_HARD_VALID_CANDIDATES",
                        }
                    ]
                }
            )
        )

    def test_plan_seed_variant_bundles_no_longer_requires_solver_output(self) -> None:
        calls: list[dict[str, object]] = []

        class _DummyConceptGenerator:
            def generate_bundle(self, **kwargs):
                calls.append(kwargs)
                return {
                    "status": "OK",
                    "room_type": "living_room",
                    "concepts": [
                        {
                            "concept_family": "focal_axis",
                            "spatial_character": "Focal axis",
                            "cluster_zone_plan": [],
                        },
                        {
                            "concept_family": "open_center",
                            "spatial_character": "Open center",
                            "cluster_zone_plan": [],
                        },
                    ],
                }

        bundle = {
            "cluster_outlines": {
                "living": {
                    "cluster_footprint": {
                        "local_bbox": {
                            "min_x": 0,
                            "min_y": 0,
                            "max_x": 900,
                            "max_y": 1200,
                        }
                    },
                    "orientation_meta": {
                        "cluster_front_local": {"dx": 0, "dy": 1},
                        "cluster_axis_local": {"dx": 1, "dy": 0},
                    },
                }
            }
        }
        room_output = {
            "room": {
                "room_id": "room_1",
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
            }
        }

        planned_bundles, payload = orchestrator._plan_seed_variant_bundles(
            bundle=bundle,
            room_output=room_output,
            guidance_text="living room",
            concept_generator=_DummyConceptGenerator(),
            target_count=2,
        )

        self.assertEqual(len(planned_bundles), 2)
        self.assertEqual(len(payload["variants"]), 2)
        seed_notes = (calls[0]["clusters_json"].get("seed_layout_state") or {}).get(
            "notes"
        ) or []
        self.assertIn("No macro seed layout has been solved yet", seed_notes[0])

    def test_solve_planned_variant_bundle_uses_branch_relation_plan(self) -> None:
        captured: dict[str, object] = {}

        class _DummySolver:
            def generate(self, **kwargs):
                captured["relation_plan_json"] = kwargs.get("relation_plan_json")
                return {
                    "status": "OK",
                    "cluster_transforms": [
                        {"cluster_id": "living", "x": 1000, "y": 900, "rot": 0}
                    ],
                    "selected_variants": [
                        {"cluster_id": "living", "variant_id": "living::base"}
                    ],
                    "notes": ["Solved concept branch."],
                }

        bundle = {
            "seed_index": 2,
            "relation_plan": {
                "status": "OK",
                "cluster_relations": [
                    {
                        "a": "living",
                        "b": "media",
                        "relation": "separate",
                    }
                ],
            },
            "cluster_outlines": {"living": {"cluster_footprint": {}}},
            "cluster_output": {"clusters": []},
            "merged_output": {"clusters": []},
            "tier_output": {"decisions": []},
            "notes": [],
        }

        solved_bundle = orchestrator._solve_planned_variant_bundle(
            bundle=bundle,
            room_output={"room": {"room_id": "room_1"}},
            guidance_text="living room",
            composer=object(),
            solver=_DummySolver(),
            max_attempts=1,
        )

        self.assertIsNotNone(solved_bundle)
        self.assertEqual(
            captured["relation_plan_json"],
            bundle["relation_plan"],
        )
        self.assertEqual(
            solved_bundle["solver_relation_plan"],
            bundle["relation_plan"],
        )

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

    def test_storage_front_token_repair_only_updates_offending_storage(self) -> None:
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

        self.assertTrue(changed)
        clusters = merged_output["clusters"]
        storage = clusters[0]
        sleep = clusters[1]
        self.assertEqual(
            storage["object_program"]["object_specs_by_id"]["wardrobe"]["front"],
            "left",
        )
        self.assertEqual(
            storage["cluster_rules"]["facing"]["wardrobe"]["front"],
            "left",
        )
        self.assertEqual(
            sleep["object_program"]["object_specs_by_id"]["bed"]["front"],
            "top",
        )

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
        self.assertLessEqual(len(solver.calls), 4)

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
