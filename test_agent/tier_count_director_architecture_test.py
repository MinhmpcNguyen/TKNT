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
            decisions_by_type["coffee_table_1"]["rep_dims_m"][
                "source_id"
            ].startswith("coffee_table_")
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
