from __future__ import annotations

import importlib
import sys
import types
import unittest

fake_base_client = types.ModuleType("clients.base_client")
fake_base_client.ChatMessage = dict
sys.modules.setdefault("clients.base_client", fake_base_client)

fake_llm_client = types.ModuleType("clients.llm_client")
fake_llm_client.get_llm_client = lambda *args, **kwargs: None
sys.modules.setdefault("clients.llm_client", fake_llm_client)

fake_gemini_config = types.ModuleType("config.gemini_config")


class _DummyGeminiConfig:
    AGENT_MODELS = {}


fake_gemini_config.GeminiConfig = _DummyGeminiConfig
sys.modules.setdefault("config.gemini_config", fake_gemini_config)

fake_schema = types.ModuleType("agent_schema.cluster_relation_planner_schema")


class _DummyPlannerOutput:
    @classmethod
    def model_validate(cls, payload):
        class _Result:
            def model_dump(self, **_kwargs):
                return payload

        return _Result()


fake_schema.ClusterRelationPlannerOutput = _DummyPlannerOutput
sys.modules.setdefault("agent_schema.cluster_relation_planner_schema", fake_schema)

fake_prompt = types.ModuleType("prompt.cluster_relation_planner")
fake_prompt.CLUSTER_RELATION_PLANNER_PROMPT = ""
sys.modules.setdefault("prompt.cluster_relation_planner", fake_prompt)

fake_system = types.ModuleType("prompt.system")
fake_system.SYSTEM_PROMPT = ""
sys.modules.setdefault("prompt.system", fake_system)

planner_mod = importlib.import_module("agent.cluster_relation_planner")


class ClusterRelationPlannerArchitectureTest(unittest.TestCase):
    def test_generate_uses_explicit_model_name_override(self) -> None:
        captured: dict[str, object] = {}

        class _DummyClient:
            def chat_completion(
                self,
                _messages,
                *,
                model_key=None,
                model_name=None,
                temperature=None,
                max_tokens=None,
            ):
                captured["model_key"] = model_key
                captured["model_name"] = model_name
                captured["temperature"] = temperature
                captured["max_tokens"] = max_tokens
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content='{"status":"OK","cluster_relations":[],"cluster_directional_relations":[]}'
                            )
                        )
                    ]
                )

        original_get_llm_client = planner_mod.get_llm_client
        try:
            planner_mod.get_llm_client = lambda provider="gemini": _DummyClient()
            planner = planner_mod.ClusterRelationPlanner()
            result = planner.generate(
                room_model_json={"room": {"room_id": "r0"}},
                clusters_json={"clusters": []},
                model_name="gemma-4-31b-it",
            )
        finally:
            planner_mod.get_llm_client = original_get_llm_client

        self.assertEqual(captured["model_key"], "primary")
        self.assertEqual(captured["model_name"], "gemma-4-31b-it")
        self.assertEqual(result["status"], "OK")

    def test_generate_falls_back_when_llm_returns_invalid_json(self) -> None:
        calls: list[object] = []

        class _InvalidJsonClient:
            def chat_completion(
                self,
                _messages,
                *,
                model_key=None,
                model_name=None,
                temperature=None,
                max_tokens=None,
            ):
                calls.append(
                    {
                        "model_key": model_key,
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content="not valid json even after retry"
                            )
                        )
                    ]
                )

        original_get_llm_client = planner_mod.get_llm_client
        try:
            planner_mod.get_llm_client = lambda provider="gemini": _InvalidJsonClient()
            planner = planner_mod.ClusterRelationPlanner()
            result = planner.generate(
                room_model_json={
                    "room": {"room_id": "r0"},
                    "meta": {"room_type": "living_room"},
                },
                clusters_json={
                    "clusters": [
                        {"cluster_id": "living", "members": ["sofa", "coffee_table"]},
                        {"cluster_id": "media", "members": ["tv_console"]},
                    ]
                },
                model_name="gemma-4-31b-it",
            )
        finally:
            planner_mod.get_llm_client = original_get_llm_client

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["room_id"], "r0")
        self.assertEqual(result["object_orientations"], [])
        self.assertTrue(
            any(
                "deterministic macro fallback" in note
                for note in result.get("notes") or []
            )
        )
        self.assertTrue(
            any(
                row.get("a") == "living"
                and row.get("b") == "media"
                and row.get("relation") == "face_each_other"
                for row in result.get("cluster_directional_relations") or []
            )
        )

    def test_merges_existing_intent_profile_without_unhashable_list_error(self) -> None:
        payload = {
            "status": "OK",
            "layout_intent_profile": {
                "primary_cluster_id": "living",
                "secondary_cluster_id": [],
            },
            "cluster_relations": [
                {"a": "living", "b": "media", "relation": "near", "priority": "high"}
            ],
            "cluster_directional_relations": [
                {
                    "a": "living",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                }
            ],
        }
        room_model = {"room": {"room_id": "r0"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {"cluster_id": "living", "members": ["sofa", "coffee_table"]},
                {"cluster_id": "media", "members": ["tv_console"]},
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        self.assertEqual(
            normalized["layout_intent_profile"]["secondary_cluster_id"], "media"
        )

    def test_derives_layout_intent_profile(self) -> None:
        payload = {
            "status": "OK",
            "cluster_relations": [
                {"a": "living", "b": "media", "relation": "near", "priority": "high"}
            ],
            "cluster_directional_relations": [
                {
                    "a": "living",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                }
            ],
            "circulation_plan": {
                "keep_open_regions": [
                    {
                        "type": "center_lane",
                        "near": "room_center",
                        "priority": "high",
                    }
                ]
            },
        }
        room_model = {"room": {"room_id": "r0"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {"cluster_id": "living", "members": ["sofa", "coffee_table"]},
                {"cluster_id": "media", "members": ["tv_console"]},
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        self.assertEqual(normalized["layout_intent_profile"]["focus_mode"], "viewing")
        self.assertEqual(
            normalized["layout_intent_profile"]["primary_cluster_id"], "living"
        )
        self.assertEqual(
            normalized["layout_intent_profile"]["center_open_preference"], "high"
        )

    def test_normalize_clears_missing_when_status_is_ok(self) -> None:
        normalized = planner_mod._normalize_relation_plan_payload(
            {
                "status": "OK",
                "missing": ["seed_variant_notes", "something_else"],
                "cluster_relations": [],
            },
            room_model_json={"room": {"room_id": "r0"}},
            clusters_json={"clusters": []},
        )

        self.assertEqual(normalized["status"], "OK")
        self.assertEqual(normalized["missing"], [])

    def test_normalization_drops_planner_object_orientations(self) -> None:
        payload = {
            "status": "OK",
            "cluster_orientations": [
                {
                    "cluster_id": "living",
                    "intents": ["face_cluster"],
                    "target_cluster_id": "media",
                }
            ],
            "object_orientations": [
                {
                    "cluster_id": "living",
                    "object_id": "armchair",
                    "intents": ["face_object"],
                    "target_object_id": "tv_console",
                }
            ],
        }
        room_model = {"room": {"room_id": "r0"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {
                    "cluster_id": "living",
                    "members": ["sectional_sofa", "armchair"],
                    "anchors": ["sectional_sofa"],
                },
                {
                    "cluster_id": "media",
                    "members": ["tv_console"],
                    "anchors": ["tv_console"],
                },
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        self.assertEqual(normalized["object_orientations"], [])

    def test_rebalances_viewing_media_affinity_away_from_window_side(self) -> None:
        payload = {
            "status": "OK",
            "cluster_affinities": [
                {
                    "cluster_id": "media",
                    "prefer": ["wall", "window_side"],
                    "avoid": [],
                    "priority": "high",
                }
            ],
            "cluster_relations": [
                {"a": "living", "b": "media", "relation": "near", "priority": "high"}
            ],
            "cluster_directional_relations": [
                {
                    "a": "living",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                }
            ],
        }
        room_model = {"room": {"room_id": "r0"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {"cluster_id": "living", "members": ["sofa", "coffee_table"]},
                {"cluster_id": "media", "members": ["tv_console"]},
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        media_affinity = next(
            row
            for row in normalized["cluster_affinities"]
            if row.get("cluster_id") == "media"
        )
        self.assertIn("wall", media_affinity["prefer"])
        self.assertIn("far_from_entry", media_affinity["prefer"])
        self.assertNotIn("window_side", media_affinity["prefer"])
        self.assertIn("center", media_affinity["avoid"])

    def test_affinity_conflict_prefers_avoid_over_prefer(self) -> None:
        payload = {
            "status": "OK",
            "cluster_affinities": [
                {
                    "cluster_id": "seating",
                    "prefer": ["center", "wall", "far_from_entry"],
                    "avoid": ["center", "window_blocking"],
                    "priority": "high",
                }
            ],
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json={"room": {"room_id": "r0"}},
            clusters_json={
                "clusters": [
                    {
                        "cluster_id": "seating",
                        "members": ["sectional_sofa", "coffee_table"],
                    }
                ]
            },
        )

        seating_affinity = normalized["cluster_affinities"][0]
        self.assertEqual(seating_affinity["cluster_id"], "seating")
        self.assertNotIn("center", seating_affinity["prefer"])
        self.assertIn("center", seating_affinity["avoid"])
        self.assertTrue(
            any(
                "overlapping cluster affinity tags" in note
                for note in normalized.get("notes") or []
            )
        )

    def test_elevates_large_primary_lounge_cluster_to_wall_anchor(self) -> None:
        payload = {"status": "OK"}
        room_model = {"room": {"room_id": "r3"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {
                    "cluster_id": "living",
                    "members": ["sectional_sofa", "coffee_table"],
                    "cluster_footprint": {
                        "rects": [{"x": 0, "y": 0, "w": 2800, "h": 1500}]
                    },
                },
                {
                    "cluster_id": "support",
                    "members": ["bookshelf"],
                    "cluster_footprint": {
                        "rects": [{"x": 0, "y": 0, "w": 900, "h": 400}]
                    },
                },
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        living_affinity = next(
            row
            for row in normalized["cluster_affinities"]
            if row.get("cluster_id") == "living"
        )
        living_orientation = next(
            row
            for row in normalized["cluster_orientations"]
            if row.get("cluster_id") == "living"
        )

        self.assertIn("wall", living_affinity["prefer"])
        self.assertIn("long_wall", living_affinity["prefer"])
        self.assertIn("back_to_wall", living_orientation["intents"])

    def test_elevates_primary_media_cluster_with_access_orientation(self) -> None:
        payload = {"status": "OK"}
        room_model = {"room": {"room_id": "r4"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {
                    "cluster_id": "media_zone",
                    "members": ["television", "media_shelf"],
                    "cluster_footprint": {
                        "rects": [{"x": 0, "y": 0, "w": 2200, "h": 700}]
                    },
                },
                {
                    "cluster_id": "support",
                    "members": ["plant"],
                    "cluster_footprint": {
                        "rects": [{"x": 0, "y": 0, "w": 500, "h": 500}]
                    },
                },
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        media_orientation = next(
            row
            for row in normalized["cluster_orientations"]
            if row.get("cluster_id") == "media_zone"
        )
        self.assertIn("back_to_wall", media_orientation["intents"])
        self.assertIn("access_to_open_space", media_orientation["intents"])

    def test_prunes_non_focal_directional_relations(self) -> None:
        payload = {
            "status": "OK",
            "cluster_directional_relations": [
                {
                    "a": "sleep",
                    "b": "living",
                    "relation": "face_each_other",
                    "priority": "high",
                }
            ],
            "cluster_relations": [
                {"a": "sleep", "b": "living", "relation": "near", "priority": "medium"}
            ],
        }
        room_model = {"room": {"room_id": "r1"}, "meta": {"room_type": "bedroom"}}
        clusters = {
            "clusters": [
                {"cluster_id": "sleep", "members": ["bed", "nightstand"]},
                {"cluster_id": "living", "members": ["armchair", "floor_lamp"]},
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        self.assertEqual(normalized["cluster_directional_relations"], [])

    def test_keeps_focal_viewing_directional_relation(self) -> None:
        payload = {
            "status": "OK",
            "cluster_directional_relations": [
                {
                    "a": "living",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                }
            ],
            "cluster_relations": [
                {"a": "living", "b": "media", "relation": "near", "priority": "high"}
            ],
        }
        room_model = {"room": {"room_id": "r2"}, "meta": {"room_type": "living_room"}}
        clusters = {
            "clusters": [
                {"cluster_id": "living", "members": ["sofa", "coffee_table"]},
                {"cluster_id": "media", "members": ["tv_console"]},
            ]
        }

        normalized = planner_mod._normalize_relation_plan_payload(
            payload,
            room_model_json=room_model,
            clusters_json=clusters,
        )

        self.assertEqual(
            normalized["cluster_directional_relations"],
            [
                {
                    "a": "living",
                    "b": "media",
                    "relation": "face_each_other",
                    "priority": "high",
                    "reason": "directional relation",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
