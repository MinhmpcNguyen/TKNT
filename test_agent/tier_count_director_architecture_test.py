from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_tier_count_director = importlib.import_module("agent.tier_count_director")
TierCountDirector = _tier_count_director.TierCountDirector
_request_contract = importlib.import_module("agent.request_contract")
_tier_count_tools = importlib.import_module("tier_count.tools")


def _sample_room() -> dict[str, Any]:
    return {
        "room": {
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ]
        },
        "obstacles": [],
        "meta": {"style": "japandi"},
    }


def _sample_clusters() -> dict[str, Any]:
    return {
        "clusters": [
            {
                "cluster_id": "sleep_area",
                "tag": "sleep",
                "members": ["bed", "nightstand"],
                "anchors": ["bed"],
                "cluster_rules": {"access_requirements": []},
            },
            {
                "cluster_id": "work_area",
                "tag": "work",
                "members": ["desk", "desk_chair"],
                "anchors": ["desk"],
                "hard_constraints": [{"type": "requires_access", "id": "desk"}],
                "cluster_rules": {"access_requirements": [{"id": "desk"}]},
            },
            {
                "cluster_id": "misc_area",
                "tag": "misc",
                "members": ["floor_lamp"],
                "anchors": ["floor_lamp"],
                "cluster_rules": {"allow_empty_cluster": True},
            },
        ]
    }


def _living_seed_clusters() -> dict[str, Any]:
    return {
        "clusters": [
            {
                "cluster_id": "seating_zone",
                "tag": "living",
                "members": [
                    "sofa",
                    "sectional_sofa",
                    "coffee_table",
                    "ottoman",
                    "recliner",
                ],
                "anchors": ["sofa", "sectional_sofa"],
                "cluster_rules": {"access_requirements": []},
            }
        ]
    }


def _make_profile(source_id: str) -> dict[str, Any]:
    return {
        "rep_dims_m": {
            "S": {
                "L": 0.4,
                "W": 0.4,
                "H": 0.5,
                "A": 0.16,
                "R": 1.0,
                "source_id": f"{source_id}_s",
            },
            "M": {
                "L": 1.0,
                "W": 0.6,
                "H": 0.8,
                "A": 0.6,
                "R": 1.67,
                "source_id": f"{source_id}_m",
            },
            "L": {
                "L": 1.6,
                "W": 0.9,
                "H": 1.0,
                "A": 1.44,
                "R": 1.78,
                "source_id": f"{source_id}_l",
            },
        }
    }


def _budget_detail(
    *,
    category: str,
    cluster_id: str,
    quantity: int,
    priority: str,
    effective_footprint_m2: float,
    request_contract_intent: str = "",
) -> dict[str, Any]:
    return {
        "decision_id": category,
        "cluster_id": cluster_id,
        "category": category,
        "size_tier": "S",
        "quantity": quantity,
        "min_quantity": 0,
        "priority": priority,
        "request_contract_intent": request_contract_intent,
        "clearance_depth_m": 0.0,
        "base_footprint_m2_per_item": effective_footprint_m2,
        "effective_footprint_m2_per_item": effective_footprint_m2,
        "base_footprint_m2_total": effective_footprint_m2 * quantity,
        "effective_footprint_m2_total": effective_footprint_m2 * quantity,
        "footprint_options_by_tier": {
            "S": {
                "base_footprint_m2_per_item": effective_footprint_m2,
                "effective_footprint_m2_per_item": effective_footprint_m2,
                "clearance_depth_m": 0.0,
            }
        },
    }


class TierCountDirectorArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_get_tool_registry = _tier_count_director._get_tool_registry

    def tearDown(self) -> None:
        _tier_count_director._get_tool_registry = self._original_get_tool_registry

    def test_generate_uses_deterministic_policy_and_alias_backfill(self) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {
                    "sleep_area": 4.5,
                    "work_area": 2.5,
                    "misc_area": 1.0,
                },
            }

        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Need a compact bedroom with a small work corner.",
            special_notes="Keep it uncluttered and airy.",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "japandi"}},
            clusters_json=_sample_clusters(),
        )

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["budget_valid"])
        self.assertIn("deterministic", result["global_notes"][0].lower())

        decisions_by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(decisions_by_type["desk_chair"]["category"], "desk_chair")
        self.assertIn("rep_dims_m", decisions_by_type["desk_chair"])
        self.assertEqual(decisions_by_type["bed"]["quantity"], 1)

    def test_generate_uses_canonical_profiles_for_instance_member_ids(self) -> None:
        requested_categories: list[str] = []

        clusters_json = {
            "clusters": [
                {
                    "cluster_id": "main_seating",
                    "tag": "living",
                    "members": [
                        "sofa_1",
                        "coffee_table_1",
                        "armchair_1",
                        "armchair_2",
                    ],
                    "anchors": ["sofa_1"],
                    "cluster_rules": {"access_requirements": []},
                }
            ]
        }

        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            requested_categories.extend(categories)
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            profiles = kwargs["size_profiles_by_category"]
            self.assertIn("sofa_1", profiles)
            self.assertIn("coffee_table_1", profiles)
            self.assertIn("armchair_1", profiles)
            self.assertIn("armchair_2", profiles)
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {"main_seating": 6.0},
            }

        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Design a practical living room.",
            special_notes="",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=clusters_json,
        )

        self.assertEqual(requested_categories, ["sofa", "coffee_table", "armchair"])
        decisions_by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(decisions_by_type["sofa_1"]["category"], "sofa_1")
        self.assertTrue(
            decisions_by_type["sofa_1"]["rep_dims_m"]["source_id"].startswith("sofa_")
        )
        self.assertTrue(
            decisions_by_type["coffee_table_1"]["rep_dims_m"]["source_id"].startswith(
                "coffee_table_"
            )
        )
        self.assertTrue(
            decisions_by_type["armchair_1"]["rep_dims_m"]["source_id"].startswith(
                "armchair_"
            )
        )

    def test_generate_applies_budget_recommendations_without_llm_loop(self) -> None:
        budget_calls: list[list[dict[str, Any]]] = []

        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            decisions = [
                dict(row) for row in kwargs["decisions"] if isinstance(row, dict)
            ]
            budget_calls.append(decisions)

            if len(budget_calls) == 1:
                recommended = [dict(row) for row in decisions]
                for row in recommended:
                    if row.get("object_type") == "floor_lamp":
                        row["quantity"] = 0
                        row["size_tier"] = "S"
                return {
                    "input_decisions_fit": False,
                    "recommended_decisions_fit": False,
                    "recommended_decisions": recommended,
                    "decision_footprint_details": [],
                    "cluster_budget_limits_m2": {
                        "sleep_area": 4.5,
                        "work_area": 2.5,
                        "misc_area": 0.0,
                    },
                }

            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": decisions,
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {
                    "sleep_area": 4.5,
                    "work_area": 2.5,
                    "misc_area": 0.0,
                },
            }

        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Need a compact bedroom with a small work corner.",
            special_notes="Keep it uncluttered and airy.",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "japandi"}},
            clusters_json=_sample_clusters(),
        )

        self.assertEqual(len(budget_calls), 2)
        self.assertEqual(result["status"], "OK")

        decisions_by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(decisions_by_type["floor_lamp"]["quantity"], 0)

    def test_budget_recommendation_keeps_small_optional_workflow_for_solver_trial(
        self,
    ) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["desk"]["rep_dims_m"]["S"] = {
                "L": 0.98,
                "W": 0.51,
                "H": 0.78,
                "A": 0.4998,
                "R": 1.92,
                "source_id": "desk_s",
            }
            profiles["chair"]["rep_dims_m"]["S"] = {
                "L": 0.45,
                "W": 0.45,
                "H": 0.85,
                "A": 0.2025,
                "R": 1.0,
                "source_id": "chair_s",
            }
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            recommended = [
                dict(row) for row in kwargs["decisions"] if isinstance(row, dict)
            ]
            for row in recommended:
                if row.get("object_type") in {"desk", "chair"}:
                    row["quantity"] = 0
                    row["size_tier"] = "S"
            return {
                "input_decisions_fit": False,
                "recommended_decisions_fit": True,
                "recommended_decisions": recommended,
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {"work_area": 0.0},
            }

        clusters_json = {
            "semantic_layout_program": {
                "active_clusters": [
                    {
                        "cluster_id": "work_area",
                        "priority": "support",
                        "required_bundles": [
                            {
                                "bundle_id": "work_area_bundle",
                                "objects": [
                                    {
                                        "object_type": "desk",
                                        "role": "workflow_anchor",
                                    },
                                    {"object_type": "chair", "role": "support"},
                                ],
                            }
                        ],
                    }
                ],
            },
            "clusters": [
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                    "cluster_rules": {"allow_empty_cluster": True},
                }
            ],
        }
        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Add a compact work corner if the bedroom has space.",
            special_notes="Keep circulation comfortable.",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=clusters_json,
        )

        decisions_by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(result["budget_mode"], "recommended_with_solver_trials")
        self.assertEqual(decisions_by_type["desk"]["quantity"], 1)
        self.assertEqual(decisions_by_type["chair"]["quantity"], 1)
        self.assertTrue(decisions_by_type["desk"]["solver_trial"])
        self.assertTrue(decisions_by_type["chair"]["trial_optional"])
        self.assertTrue(decisions_by_type["desk"]["droppable"])
        self.assertFalse(decisions_by_type["desk"]["protected"])
        self.assertEqual(
            {row["object_type"] for row in result["budget_trial_restores"]},
            {"desk", "chair"},
        )

    def test_input_fit_marks_optional_workflow_for_solver_trial(self) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["desk"]["rep_dims_m"]["S"] = {
                "L": 0.98,
                "W": 0.51,
                "H": 0.78,
                "A": 0.4998,
                "R": 1.92,
                "source_id": "desk_s",
            }
            profiles["chair"]["rep_dims_m"]["S"] = {
                "L": 0.45,
                "W": 0.45,
                "H": 0.85,
                "A": 0.2025,
                "R": 1.0,
                "source_id": "chair_s",
            }
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {"work_area": 2.0},
            }

        clusters_json = {
            "semantic_layout_program": {
                "active_clusters": [
                    {
                        "cluster_id": "work_area",
                        "layout_role": "optional",
                        "priority": "support",
                        "required_bundles": [
                            {
                                "bundle_id": "work_area_bundle",
                                "objects": [
                                    {
                                        "object_type": "desk",
                                        "role": "workflow_anchor",
                                    },
                                    {"object_type": "chair", "role": "support"},
                                ],
                            }
                        ],
                    }
                ],
            },
            "clusters": [
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                    "cluster_rules": {"allow_empty_cluster": True},
                }
            ],
        }
        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Add a compact work corner if the bedroom has space.",
            special_notes="Keep circulation comfortable.",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=clusters_json,
        )

        decisions_by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(result["budget_mode"], "input")
        self.assertTrue(decisions_by_type["desk"]["solver_trial"])
        self.assertTrue(decisions_by_type["chair"]["trial_optional"])
        self.assertTrue(decisions_by_type["desk"]["droppable"])
        self.assertFalse(decisions_by_type["desk"]["protected"])
        self.assertEqual(
            {row["object_type"] for row in result["solver_trial_optionals"]},
            {"desk", "chair"},
        )

    def test_generate_protects_explicit_request_contract_objects(self) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {"work_area": 4.0},
            }

        semantic_program = (
            _request_contract.attach_request_contract_to_semantic_program(
                {
                    "active_clusters": [
                        {
                            "cluster_id": "work_area",
                            "priority": "support",
                            "required_bundles": [
                                {
                                    "bundle_id": "workstation",
                                    "objects": [
                                        {"object_type": "desk", "role": "support"},
                                        {"object_type": "chair", "role": "support"},
                                        {
                                            "object_type": "bookshelf",
                                            "role": "support",
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                },
                brief_text="Include a compact desk with a chair and a bookshelf.",
            )
        )
        clusters_json = {
            "semantic_layout_program": semantic_program,
            "clusters": [
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "chair", "bookshelf"],
                    "anchors": ["desk"],
                    "cluster_rules": {"allow_empty_cluster": True},
                }
            ],
        }
        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Include a compact desk with a chair and a bookshelf.",
            special_notes="",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=clusters_json,
        )

        decisions_by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        for object_type in ("desk", "chair", "bookshelf"):
            self.assertGreaterEqual(decisions_by_type[object_type]["quantity"], 1)
            self.assertEqual(
                decisions_by_type[object_type]["request_contract_intent"],
                "must_keep",
            )
        self.assertEqual(decisions_by_type["chair"]["priority"], "primary")
        self.assertEqual(decisions_by_type["bookshelf"]["priority"], "primary")

    def test_generate_logs_request_contract_degradation(self) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            decisions = [
                dict(row) for row in kwargs["decisions"] if isinstance(row, dict)
            ]
            recommended = [dict(row) for row in decisions]
            for row in recommended:
                if row.get("object_type") == "chair":
                    row["quantity"] = 0
            return {
                "input_decisions_fit": False,
                "recommended_decisions_fit": True,
                "recommended_decisions": recommended,
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {"work_area": 1.0},
            }

        semantic_program = (
            _request_contract.attach_request_contract_to_semantic_program(
                {
                    "active_clusters": [
                        {
                            "cluster_id": "work_area",
                            "priority": "support",
                            "required_bundles": [
                                {
                                    "bundle_id": "workstation",
                                    "objects": [
                                        {"object_type": "desk", "role": "support"},
                                        {"object_type": "chair", "role": "support"},
                                    ],
                                }
                            ],
                        }
                    ],
                },
                brief_text="Include a compact desk with a chair.",
            )
        )
        clusters_json = {
            "semantic_layout_program": semantic_program,
            "clusters": [
                {
                    "cluster_id": "work_area",
                    "tag": "work",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                    "cluster_rules": {"allow_empty_cluster": True},
                }
            ],
        }
        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Include a compact desk with a chair.",
            special_notes="",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=clusters_json,
        )

        report = result["request_contract_degradation_report"]
        self.assertEqual(result["degradation_status"], "DEGRADED_OK")
        self.assertEqual(report["dropped"][0]["object_type"], "chair")
        self.assertEqual(
            report["dropped"][0]["degradation_reason"],
            "dropped_after_budget_repair",
        )

    def test_target_if_viable_request_contract_does_not_hard_protect_support(
        self,
    ) -> None:
        contract = _request_contract.sanitize_request_contract(
            {
                "objects": [
                    {
                        "object_type": "armchair",
                        "intent": "target_if_viable",
                        "min_keep": 0,
                        "target_count": 2,
                        "max_keep": 2,
                        "evidence": "one or two armchairs if possible",
                        "reason": "fit dependent",
                        "confidence": 0.9,
                    }
                ],
                "groups": [],
            },
            brief_text="Add one or two armchairs if possible.",
            available_object_types=["armchair"],
        )

        obj = _tier_count_director._apply_request_contract_to_object(
            {
                "object_type": "armchair",
                "base_type": "armchair",
                "role": "support",
                "required": True,
                "protected": True,
                "droppable": False,
                "max_keep": 1,
                "min_keep": 1,
                "tier_count_explicit": True,
                "keep_if_space_surplus": False,
                "space_surplus_threshold": 0.45,
                "drop_order_bias": "neutral",
                "preserve_level": "medium",
            },
            request_contract=contract,
        )

        self.assertEqual(obj["min_keep"], 0)
        self.assertEqual(obj["max_keep"], 2)
        self.assertFalse(obj["required"])
        self.assertFalse(obj["protected"])
        self.assertTrue(obj["droppable"])
        self.assertTrue(obj["keep_if_space_surplus"])

    def test_target_if_viable_support_gets_primary_budget_priority(self) -> None:
        priority = _tier_count_director._decision_priority(
            obj={
                "object_type": "nightstand",
                "request_contract_intent": "target_if_viable",
                "min_keep": 0,
            },
            role="support",
            bundle={},
        )

        self.assertEqual(priority, "primary")

    def test_quantity_target_uses_contract_count_without_object_name_branch(
        self,
    ) -> None:
        quantity = _tier_count_director._select_quantity(
            obj={
                "object_type": "support_surface",
                "base_type": "support_surface",
                "role": "support",
                "required": True,
                "protected": True,
                "min_keep": 1,
                "max_keep": 2,
                "tier_count_explicit": True,
                "preserve_level": "highest",
                "request_contract_target_count": 2,
            },
            bundle={"bundle_class": "strong_support"},
            score={"total": 2.5, "circulation_penalty": 0.0, "clutter_penalty": 0.0},
            used_footprint_m2=0.5,
            capacity_model={"available_area_m2": 14.0, "clutter_budget_m2": 6.0},
            size_profiles_by_category={"support_surface": _make_profile("support")},
            room_scale="medium",
            furnishing_mode="neutral",
        )

        self.assertEqual(quantity, 2)

    def test_room_fit_backoff_keeps_area_floor_and_request_minimums(self) -> None:
        draft = {
            "status": "OK",
            "global_notes": [],
            "capacity_model": {"available_area_m2": 11.5, "clutter_budget_m2": 4.0},
            "decisions": [
                {
                    "object_type": "anchor_surface",
                    "category": "anchor_surface",
                    "cluster_id": "core",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "anchor",
                    "role": "dominant_anchor",
                    "min_keep": 1,
                    "preserve_level": "highest",
                    "drop_order_bias": "drop_last",
                    "rep_dims_m": {"L": 2.0, "W": 1.5, "A": 3.0},
                },
                {
                    "object_type": "work_surface",
                    "category": "work_surface",
                    "cluster_id": "work",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "primary",
                    "role": "workflow_anchor",
                    "min_keep": 1,
                    "preserve_level": "high",
                    "drop_order_bias": "drop_last",
                    "rep_dims_m": {"L": 1.0, "W": 0.8, "A": 0.8},
                },
                {
                    "object_type": "seat",
                    "category": "seat",
                    "cluster_id": "work",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "primary",
                    "role": "support",
                    "min_keep": 1,
                    "preserve_level": "high",
                    "drop_order_bias": "drop_last",
                    "rep_dims_m": {"L": 0.6, "W": 0.6, "A": 0.36},
                },
                {
                    "object_type": "support_surface",
                    "category": "support_surface",
                    "cluster_id": "core",
                    "quantity": 2,
                    "size_tier": "M",
                    "priority": "primary",
                    "role": "support",
                    "min_keep": 1,
                    "preserve_level": "highest",
                    "drop_order_bias": "drop_last",
                    "request_contract_intent": "must_keep",
                    "rep_dims_m": {"L": 0.7, "W": 0.5, "A": 0.35},
                },
                {
                    "object_type": "secondary_storage",
                    "category": "secondary_storage",
                    "cluster_id": "storage",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "secondary",
                    "role": "support",
                    "min_keep": 0,
                    "preserve_level": "medium",
                    "drop_order_bias": "neutral",
                    "rep_dims_m": {"L": 1.2, "W": 0.8, "A": 0.96},
                },
            ],
        }

        repaired = _tier_count_director._repair_overfull_draft_if_needed(
            draft,
            capacity_model=draft["capacity_model"],
            size_profiles_by_category={},
        )
        by_type = {row["object_type"]: row for row in repaired["decisions"]}

        self.assertEqual(by_type["secondary_storage"]["quantity"], 0)
        self.assertEqual(by_type["support_surface"]["quantity"], 1)
        self.assertEqual(repaired["decision_summary"]["selected_object_count"], 4)

    def test_surplus_functional_support_can_recover_from_zero_cap(self) -> None:
        cluster_hints = {
            "keep_if_space_surplus": True,
            "space_surplus_threshold": 0.5,
            "object_hints_by_type": {
                "bookshelf": {
                    "object_type": "bookshelf",
                    "min_keep": 0,
                    "max_keep": 0,
                    "keep_if_space_surplus": False,
                    "space_surplus_threshold": 0.5,
                    "drop_order_bias": "drop_first",
                    "preserve_level": "medium",
                    "preferred_size_tier": "S",
                },
                "floor_lamp": {
                    "object_type": "floor_lamp",
                    "min_keep": 0,
                    "max_keep": 0,
                    "keep_if_space_surplus": False,
                    "space_surplus_threshold": 0.5,
                    "drop_order_bias": "drop_first",
                    "preserve_level": "low",
                    "preferred_size_tier": "S",
                },
            },
        }

        bookshelf = _tier_count_director._tier_count_object_hint(
            member="bookshelf",
            cluster_hints=cluster_hints,
        )
        floor_lamp = _tier_count_director._tier_count_object_hint(
            member="floor_lamp",
            cluster_hints=cluster_hints,
        )

        self.assertEqual(bookshelf["max_keep"], 1)
        self.assertTrue(bookshelf["keep_if_space_surplus"])
        self.assertEqual(bookshelf["drop_order_bias"], "neutral")
        self.assertEqual(floor_lamp["max_keep"], 0)
        self.assertFalse(floor_lamp["keep_if_space_surplus"])

    def test_budget_repair_drops_unrequested_droppable_anchor_before_target_support(
        self,
    ) -> None:
        details = [
            _budget_detail(
                category="bed",
                cluster_id="sleep_core",
                quantity=1,
                priority="anchor",
                effective_footprint_m2=3.0,
                request_contract_intent="must_keep",
            ),
            _budget_detail(
                category="nightstand",
                cluster_id="sleep_core",
                quantity=2,
                priority="primary",
                effective_footprint_m2=0.3,
                request_contract_intent="target_if_viable",
            ),
            _budget_detail(
                category="desk",
                cluster_id="optional_activity",
                quantity=1,
                priority="anchor",
                effective_footprint_m2=0.8,
            ),
            _budget_detail(
                category="chair",
                cluster_id="optional_activity",
                quantity=1,
                priority="secondary",
                effective_footprint_m2=0.4,
            ),
        ]
        clusters_json = {
            "clusters": [
                {
                    "cluster_id": "sleep_core",
                    "tag": "sleep",
                    "members": ["bed", "nightstand"],
                    "anchors": ["bed"],
                },
                {
                    "cluster_id": "optional_activity",
                    "tag": "activity",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                    "cluster_rules": {"allow_empty_cluster": True},
                },
            ]
        }

        fitted, _cluster_totals, global_total = (
            _tier_count_tools._fit_decisions_to_budget(
                details,
                cluster_budget_limits_m2={"sleep_core": 3.6, "optional_activity": 0.0},
                global_budget_m2=3.6,
                clusters_json=clusters_json,
            )
        )
        by_category = {row["category"]: row for row in fitted}

        self.assertEqual(by_category["nightstand"]["recommended_quantity"], 2)
        self.assertEqual(by_category["desk"]["recommended_quantity"], 0)
        self.assertEqual(by_category["chair"]["recommended_quantity"], 0)
        self.assertAlmostEqual(global_total, 3.6)

    def test_budget_rebalances_after_dropping_optional_cluster(self) -> None:
        room_model = {
            "room": {
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 5000, "y": 0},
                    {"x": 5000, "y": 2000},
                    {"x": 0, "y": 2000},
                ]
            },
            "obstacles": [],
        }
        clusters_json = {
            "clusters": [
                {
                    "cluster_id": "sleep_core",
                    "members": ["bed", "nightstand"],
                    "anchors": ["bed"],
                    "cluster_rules": {"access_requirements": []},
                },
                {
                    "cluster_id": "storage_core",
                    "members": ["wardrobe"],
                    "anchors": ["wardrobe"],
                    "cluster_rules": {"access_requirements": []},
                },
                {
                    "cluster_id": "optional_activity",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                    "cluster_rules": {
                        "allow_empty_cluster": True,
                        "access_requirements": [],
                    },
                },
            ]
        }
        decisions = [
            {
                "object_type": "bed",
                "category": "bed",
                "cluster_id": "sleep_core",
                "quantity": 1,
                "size_tier": "S",
                "priority": "anchor",
                "min_keep": 1,
                "request_contract_intent": "must_keep",
            },
            {
                "object_type": "nightstand",
                "category": "nightstand",
                "cluster_id": "sleep_core",
                "quantity": 2,
                "size_tier": "S",
                "priority": "primary",
                "min_keep": 0,
                "request_contract_intent": "target_if_viable",
            },
            {
                "object_type": "wardrobe",
                "category": "wardrobe",
                "cluster_id": "storage_core",
                "quantity": 1,
                "size_tier": "S",
                "priority": "anchor",
                "min_keep": 1,
                "request_contract_intent": "must_keep",
            },
            {
                "object_type": "desk",
                "category": "desk",
                "cluster_id": "optional_activity",
                "quantity": 1,
                "size_tier": "S",
                "priority": "anchor",
                "min_keep": 0,
            },
            {
                "object_type": "chair",
                "category": "chair",
                "cluster_id": "optional_activity",
                "quantity": 1,
                "size_tier": "S",
                "priority": "secondary",
                "min_keep": 0,
            },
        ]
        areas_m2 = {
            "bed": 3.0,
            "nightstand": 0.4,
            "wardrobe": 1.0,
            "desk": 1.5,
            "chair": 0.3,
        }
        size_profiles = {
            category: {
                "rep_dims_m": {
                    tier: {"L": area, "W": 1.0, "H": 1.0} for tier in ("S", "M", "L")
                }
            }
            for category, area in areas_m2.items()
        }

        result = _tier_count_tools.estimate_budget(
            room_model=room_model,
            decisions=decisions,
            size_profiles_by_category=size_profiles,
            style="",
            user_notes="",
            ratio_override=0.48,
            clusters_json=clusters_json,
            rescue_mode=False,
        )
        by_type = {
            str(row["object_type"]): row for row in result["recommended_decisions"]
        }

        self.assertEqual(by_type["nightstand"]["quantity"], 2)
        self.assertEqual(by_type["desk"]["quantity"], 0)
        self.assertEqual(by_type["chair"]["quantity"], 0)
        self.assertEqual(result["removed_clusters_after_fit"], ["optional_activity"])

    def test_extract_anchors_by_cluster_ignores_anchors_outside_members(self) -> None:
        clusters_json = {
            "clusters": [
                {
                    "cluster_id": "living_storage",
                    "members": ["tv_console", "storage_cabinet"],
                    "anchors": ["sofa"],
                }
            ]
        }

        anchors_by_cluster = _tier_count_director._extract_anchors_by_cluster(
            clusters_json
        )
        members_by_cluster = _tier_count_director._extract_members_by_cluster(
            clusters_json
        )
        ok, detail = _tier_count_director._validate_decisions(
            [
                {
                    "object_type": "tv_console",
                    "category": "tv_console",
                    "cluster_id": "living_storage",
                    "quantity": 1,
                    "size_tier": "L",
                    "priority": "secondary",
                },
                {
                    "object_type": "storage_cabinet",
                    "category": "storage_cabinet",
                    "cluster_id": "living_storage",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "anchor",
                },
            ],
            required_types=[],
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
        )

        self.assertEqual(anchors_by_cluster["living_storage"], set())
        self.assertTrue(ok, detail)

    def test_validate_decisions_allows_same_type_in_multiple_clusters(self) -> None:
        clusters_json = {
            "clusters": [
                {
                    "cluster_id": "work_zone",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                },
                {
                    "cluster_id": "storage_zone",
                    "members": ["wardrobe", "chair"],
                    "anchors": ["wardrobe"],
                },
            ]
        }

        anchors_by_cluster = _tier_count_director._extract_anchors_by_cluster(
            clusters_json
        )
        members_by_cluster = _tier_count_director._extract_members_by_cluster(
            clusters_json
        )
        ok, detail = _tier_count_director._validate_decisions(
            [
                {
                    "object_type": "desk",
                    "category": "desk",
                    "cluster_id": "work_zone",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "anchor",
                },
                {
                    "object_type": "chair",
                    "category": "chair",
                    "cluster_id": "work_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                },
                {
                    "object_type": "wardrobe",
                    "category": "wardrobe",
                    "cluster_id": "storage_zone",
                    "quantity": 1,
                    "size_tier": "L",
                    "priority": "anchor",
                },
                {
                    "object_type": "chair",
                    "category": "chair",
                    "cluster_id": "storage_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                },
            ],
            required_types=["desk", "chair", "wardrobe"],
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
        )

        self.assertTrue(ok, detail)

    def test_validate_decisions_requires_each_cluster_member_once(self) -> None:
        clusters_json = {
            "clusters": [
                {
                    "cluster_id": "work_zone",
                    "members": ["desk", "chair"],
                    "anchors": ["desk"],
                },
                {
                    "cluster_id": "storage_zone",
                    "members": ["wardrobe", "chair"],
                    "anchors": ["wardrobe"],
                },
            ]
        }

        anchors_by_cluster = _tier_count_director._extract_anchors_by_cluster(
            clusters_json
        )
        members_by_cluster = _tier_count_director._extract_members_by_cluster(
            clusters_json
        )
        ok, detail = _tier_count_director._validate_decisions(
            [
                {
                    "object_type": "desk",
                    "category": "desk",
                    "cluster_id": "work_zone",
                    "quantity": 1,
                    "size_tier": "M",
                    "priority": "anchor",
                },
                {
                    "object_type": "chair",
                    "category": "chair",
                    "cluster_id": "work_zone",
                    "quantity": 1,
                    "size_tier": "S",
                    "priority": "secondary",
                },
                {
                    "object_type": "wardrobe",
                    "category": "wardrobe",
                    "cluster_id": "storage_zone",
                    "quantity": 1,
                    "size_tier": "L",
                    "priority": "anchor",
                },
            ],
            required_types=["desk", "chair", "wardrobe"],
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
        )

        self.assertFalse(ok)
        self.assertIn("missing members for storage_zone", detail)

    def test_generate_collapses_multi_anchor_seed_for_irregular_minimal_room(
        self,
    ) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {
                    "seating_zone": 6.0,
                },
            }

        irregular_room = {
            "room": {
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 3000},
                    {"x": 3500, "y": 3000},
                    {"x": 3500, "y": 4000},
                    {"x": 0, "y": 4000},
                ]
            },
            "obstacles": [],
            "meta": {"style": "modern"},
        }

        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description="Design a minimal uncluttered living room.",
            special_notes="Keep it airy and open.",
            room_model_json=irregular_room,
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=_living_seed_clusters(),
        )

        self.assertEqual(result["status"], "OK")
        by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        active_anchors = [
            object_type
            for object_type in ("sofa", "sectional_sofa")
            if int(by_type[object_type]["quantity"]) > 0
        ]
        self.assertEqual(len(active_anchors), 1)
        self.assertEqual(by_type["coffee_table"]["quantity"], 0)
        self.assertEqual(by_type["ottoman"]["quantity"], 0)

    def test_generate_caps_sofa_family_when_brief_requests_single_sofa(self) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {
                    "seating_zone": 6.0,
                },
            }

        brief = (
            "Include a sofa facing a TV console, a coffee table, and one or two "
            "armchairs if they fit while keeping the room easy to move through."
        )
        clusters_json = _living_seed_clusters()
        clusters_json["request_contract"] = _request_contract.build_request_contract(
            brief_text=brief,
            available_object_types=[
                "sofa",
                "sectional_sofa",
                "coffee_table",
                "armchair",
            ],
        )

        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        result = TierCountDirector().generate(
            description=brief,
            special_notes="",
            room_model_json=_sample_room(),
            user_intent_json={"user_input": {"style": "modern"}},
            clusters_json=clusters_json,
        )

        self.assertEqual(result["status"], "OK")
        by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(by_type["sofa"]["quantity"], 1)
        self.assertEqual(by_type["sectional_sofa"]["quantity"], 0)
        self.assertEqual(by_type["sectional_sofa"]["exclusive_family"], "primary_sofa")
        self.assertEqual(
            by_type["sectional_sofa"]["exclusive_family_winner"],
            "sofa",
        )

    def test_fast_layout_mode_removes_support_items_for_primary_seed(self) -> None:
        def fake_get_size_profiles(
            *, categories: list[str], tenant_id: str | None = None
        ) -> dict[str, Any]:
            _ = tenant_id
            profiles = {category: _make_profile(category) for category in categories}
            profiles["__generic__"] = _make_profile("generic")
            return {"size_profiles_by_category": profiles, "missing_categories": []}

        def fake_estimate_budget(**kwargs: Any) -> dict[str, Any]:
            return {
                "input_decisions_fit": True,
                "recommended_decisions_fit": True,
                "recommended_decisions": kwargs["decisions"],
                "decision_footprint_details": [],
                "cluster_budget_limits_m2": {
                    "seating_zone": 6.0,
                },
            }

        irregular_room = {
            "room": {
                "polygon_ccw": [
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 3000},
                    {"x": 3500, "y": 3000},
                    {"x": 3500, "y": 4000},
                    {"x": 0, "y": 4000},
                ]
            },
            "obstacles": [],
            "meta": {"style": "modern"},
        }

        _tier_count_director._get_tool_registry = lambda: {
            "get_size_profiles": fake_get_size_profiles,
            "estimate_budget": fake_estimate_budget,
        }

        previous = os.environ.get("TKNT_LAYOUT_SPEED_MODE")
        os.environ["TKNT_LAYOUT_SPEED_MODE"] = "fast"
        try:
            result = TierCountDirector().generate(
                description="Design a practical living room that should solve quickly.",
                special_notes="Keep it airy and open.",
                room_model_json=irregular_room,
                user_intent_json={"user_input": {"style": "modern"}},
                clusters_json=_living_seed_clusters(),
            )
        finally:
            if previous is None:
                os.environ.pop("TKNT_LAYOUT_SPEED_MODE", None)
            else:
                os.environ["TKNT_LAYOUT_SPEED_MODE"] = previous

        self.assertEqual(result["status"], "OK")
        by_type = {
            row["object_type"]: row
            for row in result["decisions"]
            if isinstance(row, dict) and isinstance(row.get("object_type"), str)
        }
        self.assertEqual(by_type["coffee_table"]["quantity"], 0)
        self.assertEqual(by_type["ottoman"]["quantity"], 0)
        self.assertEqual(by_type["recliner"]["quantity"], 0)


if __name__ == "__main__":
    unittest.main()
