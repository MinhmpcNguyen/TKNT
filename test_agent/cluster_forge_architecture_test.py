from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_clients_package = types.ModuleType("clients")
fake_clients_package.__path__ = []  # type: ignore[attr-defined]
sys.modules["clients"] = fake_clients_package

fake_base_client_module = types.ModuleType("clients.base_client")


class _FakeChatMessage(TypedDict, total=False):
    role: str
    content: str


fake_base_client_module.ChatMessage = _FakeChatMessage
sys.modules["clients.base_client"] = fake_base_client_module

fake_llm_module = types.ModuleType("clients.llm_client")
fake_llm_module.get_llm_client = lambda: None
sys.modules["clients.llm_client"] = fake_llm_module

_cluster_forge = importlib.import_module("agent.cluster_forge")
ClusterForgeOutput = importlib.import_module(
    "agent_schema.clusterF_schema"
).ClusterForgeOutput
GLOBAL_LAYOUT_GRID_MM = importlib.import_module(
    "layout.grid_policy"
).GLOBAL_LAYOUT_GRID_MM


class ClusterForgeArchitectureTest(unittest.TestCase):
    def test_normalize_coerces_string_notes_and_missing_to_lists(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "sleep_area",
                    "tag": "sleep",
                    "cluster_rules": {},
                }
            ],
            "notes": "",
            "missing": "need more context",
        }

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)

        self.assertEqual(parsed.notes, [])
        self.assertEqual(parsed.missing, ["need more context"])
        self.assertEqual(
            parsed.clusters[0].cluster_rules.grid_mm,
            GLOBAL_LAYOUT_GRID_MM,
        )

    def test_normalize_recovers_common_constraint_shape_errors(self) -> None:
        payload = {
            "status": "OK",
            "clusters": [
                {
                    "cluster_id": "sleep_area",
                    "tag": "sleep",
                    "members": ["bed", "nightstand"],
                    "anchors": ["bed"],
                    "hard_constraints": [
                        {
                            "type": "contain_in",
                            "a": "nightstand",
                            "b": "bed",
                        },
                        {
                            "type": "dock_to_edge",
                            "a": "nightstand",
                            "b": "bed",
                            "b_edge": "foot",
                            "gap_min": 5,
                            "gap_max": 10,
                        },
                    ],
                    "soft_constraints": [
                        {
                            "type": "prefer_near",
                            "a": "nightstand",
                            "b": "bed",
                        }
                    ],
                    "cluster_rules": {},
                }
            ],
            "notes": [],
            "missing": [],
        }

        normalized = _cluster_forge._normalize_cluster_forge_payload(payload)
        parsed = ClusterForgeOutput.model_validate(normalized)

        dock = parsed.clusters[0].hard_constraints[0]
        near = parsed.clusters[0].soft_constraints[0]

        self.assertEqual(dock.type, "dock_to_edge")
        self.assertEqual(dock.b_edge, "back")
        self.assertEqual(dock.span, "any")
        self.assertEqual(dock.gap_min, 0)
        self.assertEqual(dock.gap_max, GLOBAL_LAYOUT_GRID_MM)
        self.assertEqual(near.type, "prefer_near")
        self.assertEqual(near.weight, 5)
        self.assertTrue(
            any("contain_in between floor objects" in note for note in parsed.notes)
        )


if __name__ == "__main__":
    unittest.main()
