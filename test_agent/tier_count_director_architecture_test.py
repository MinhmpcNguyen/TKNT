from __future__ import annotations

import importlib
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


if __name__ == "__main__":
    unittest.main()
