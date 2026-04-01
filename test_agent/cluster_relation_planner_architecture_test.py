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
