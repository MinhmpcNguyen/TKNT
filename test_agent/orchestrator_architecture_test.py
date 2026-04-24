from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


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
    "agent.cluster_relation_planner",
    "agent.phase2_controller",
    "agent.room_interpreter",
    "agent.solver.solver",
    "agent.stylist",
    "agent.tier_count_director",
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
_stub_module("agent.cluster_relation_planner", ClusterRelationPlanner=object)
_stub_module(
    "agent.phase2_controller",
    Phase2Controller=object,
    build_phase2_preview_candidate=lambda **kwargs: {},
)
_stub_module("agent.room_interpreter", RoomInterpreter=object)
_stub_module("agent.solver.solver", MacroClusterSolver=object)
_stub_module(
    "agent.stylist",
    Stylist=object,
    stylist_model_name_for_variant=lambda variant_index: "gemma-4-31b-it",
)
_stub_module("agent.tier_count_director", TierCountDirector=object)
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
        _FakeBrief("seed_stable", "gemma-4-31b-it"),
        _FakeBrief("seed_open_center", "gemini-3.1-flash-lite-preview"),
    ][: max(1, int(target_count))],
)

orchestrator = _load_module(
    "pipeline.orchestrator_for_architecture_test",
    "pipeline/orchestrator.py",
)

for _module_name in _STUBBED_MODULE_NAMES:
    sys.modules.pop(_module_name, None)


class OrchestratorArchitectureTest(unittest.TestCase):
    def test_select_final_styling_candidates_caps_to_selected_top_five(self) -> None:
        candidates = [
            {
                "source": f"candidate_{index}",
                "absolute_layout": {"objects": [], "room": {}, "status": "OK"},
            }
            for index in range(1, 10)
        ]

        original_selector = orchestrator.select_distinct_final_gallery_candidates
        orchestrator.select_distinct_final_gallery_candidates = (
            lambda **kwargs: list((kwargs.get("candidates") or [])[:5])
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

    def test_select_final_styling_candidates_falls_back_to_prefix_when_selector_empty(
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
        orchestrator.select_distinct_final_gallery_candidates = (
            lambda **_kwargs: []
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

        class _DummyRelationPlanner:
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
            relation_planner=_DummyRelationPlanner(),
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


if __name__ == "__main__":
    unittest.main()
