from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_request_contract = importlib.import_module("agent.request_contract")
attach_request_contract_to_semantic_program = (
    _request_contract.attach_request_contract_to_semantic_program
)
build_request_contract = _request_contract.build_request_contract
contract_item_for_object_type = _request_contract.contract_item_for_object_type
missing_non_functional_contract_items = (
    _request_contract.missing_non_functional_contract_items
)
sanitize_request_contract = _request_contract.sanitize_request_contract


class RequestContractTest(unittest.TestCase):
    def test_attach_keeps_llm_empty_contract_when_request_contract_llm_enabled(
        self,
    ) -> None:
        semantic_program = {
            "active_clusters": [{"cluster_id": "work", "members": ["desk"]}],
            "request_contract": {
                "source": "llm_request_contract",
                "objects": [],
                "groups": [],
            },
        }

        with patch.dict("os.environ", {"TKNT_REQUEST_CONTRACT_LLM": "1"}):
            result = attach_request_contract_to_semantic_program(
                semantic_program,
                brief_text="Include a desk.",
            )

        contract = result["request_contract"]
        self.assertEqual(contract["source"], "llm_request_contract")
        self.assertEqual(contract["objects"], [])

    def test_attach_skips_heuristic_when_request_contract_llm_enabled_and_missing(
        self,
    ) -> None:
        semantic_program = {
            "active_clusters": [{"cluster_id": "work", "members": ["desk"]}]
        }

        with patch.dict("os.environ", {"TKNT_REQUEST_CONTRACT_LLM": "1"}):
            result = attach_request_contract_to_semantic_program(
                semantic_program,
                brief_text="Include a desk.",
            )

        contract = result["request_contract"]
        self.assertEqual(contract["source"], "llm_request_contract_missing")
        self.assertEqual(contract["objects"], [])

    def test_fit_dependent_count_is_viable_target_not_required_minimum(self) -> None:
        contract = build_request_contract(
            brief_text="Include a bed with two nightstands if they fit.",
            available_object_types=["bed", "nightstand"],
        )

        objects = {
            row["object_type"]: row
            for row in contract["objects"]
            if isinstance(row, dict)
        }
        self.assertEqual(objects["nightstand"]["intent"], "target_if_viable")
        self.assertEqual(objects["nightstand"]["min_keep"], 0)
        self.assertEqual(objects["nightstand"]["target_count"], 2)

    def test_llm_target_if_viable_floor_is_preserved_by_sanitizer(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": "nightstand",
                        "intent": "target_if_viable",
                        "min_keep": 1,
                        "target_count": 2,
                        "max_keep": 2,
                        "evidence": "one or two nightstands",
                        "reason": "bounded range has a required lower bound",
                        "confidence": 0.9,
                    }
                ],
                "groups": [],
            },
            brief_text="Include one or two nightstands.",
            available_object_types=["nightstand"],
        )

        item = contract["objects"][0]
        self.assertEqual(item["intent"], "target_if_viable")
        self.assertEqual(item["min_keep"], 1)
        self.assertEqual(item["target_count"], 2)
        self.assertEqual(item["max_keep"], 2)

    def test_llm_target_if_viable_zero_floor_is_preserved_by_sanitizer(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": "armchair",
                        "intent": "target_if_viable",
                        "min_keep": 0,
                        "target_count": 2,
                        "max_keep": 2,
                        "evidence": "one or two armchairs if possible",
                        "reason": "fit dependent target",
                        "confidence": 0.9,
                    }
                ],
                "groups": [],
            },
            brief_text="Add one or two armchairs if possible.",
            available_object_types=["armchair"],
        )

        item = contract["objects"][0]
        self.assertEqual(item["intent"], "target_if_viable")
        self.assertEqual(item["min_keep"], 0)
        self.assertEqual(item["target_count"], 2)
        self.assertEqual(item["max_keep"], 2)

    def test_optional_surplus_wording_does_not_protect_object(self) -> None:
        contract = build_request_contract(
            brief_text="Add a small desk only if there is enough space.",
            available_object_types=["desk"],
        )

        item = contract["objects"][0]
        self.assertEqual(item["object_type"], "desk")
        self.assertEqual(item["intent"], "optional_if_surplus")
        self.assertEqual(item["min_keep"], 0)

    def test_heuristic_does_not_hard_code_functional_pair_groups(self) -> None:
        contract = build_request_contract(
            brief_text="Include a compact desk with a chair and a bookshelf.",
            available_object_types=["desk", "chair", "bookshelf"],
        )

        self.assertEqual(contract["groups"], [])

    def test_side_table_program_type_does_not_consume_nightstand_contract(
        self,
    ) -> None:
        contract = build_request_contract(
            brief_text="Include one or two nightstands.",
            available_object_types=["nightstand", "side_table"],
        )

        self.assertIsNone(contract_item_for_object_type(contract, "side_table"))
        self.assertIsNotNone(contract_item_for_object_type(contract, "nightstand"))

    def test_kitchen_synonyms_are_canonicalized_without_functional_pair_groups(
        self,
    ) -> None:
        contract = build_request_contract(
            brief_text=(
                "Include a refrigerator, kitchen sink, stove, prep counter, "
                "and a range hood."
            ),
            available_object_types=[
                "fridge",
                "sink",
                "stove",
                "kitchen_base_cabinet",
                "range_hood",
            ],
        )

        objects = {
            row["object_type"]: row
            for row in contract["objects"]
            if isinstance(row, dict)
        }
        self.assertEqual(
            set(objects),
            {"fridge", "kitchen_base_cabinet", "range_hood", "sink", "stove"},
        )
        self.assertEqual(objects["range_hood"]["min_keep"], 1)
        self.assertEqual(contract["groups"], [])

    def test_missing_non_functional_contract_items_reports_soft_layout_tokens(
        self,
    ) -> None:
        contract = build_request_contract(
            brief_text="Include a bed, a rug, and a ceiling lamp.",
            available_object_types=["bed"],
        )

        missing = missing_non_functional_contract_items(
            contract=contract,
            objects=[{"object_type": "bed"}],
        )

        self.assertEqual(
            [row["object_type"] for row in missing],
            ["ceiling_light", "rug"],
        )

    def test_llm_contract_restores_missing_non_functional_requested_objects(
        self,
    ) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": "bed",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": "a comfortable bed",
                        "reason": "explicitly requested",
                        "confidence": 0.9,
                    }
                ],
                "groups": [],
            },
            brief_text="Include a comfortable bed, a rug, and a ceiling lamp.",
            available_object_types=["bed"],
        )

        objects = {row["object_type"]: row for row in contract["objects"]}
        self.assertIn("rug", objects)
        self.assertIn("ceiling_light", objects)
        self.assertEqual(objects["rug"]["target_count"], 1)
        self.assertEqual(objects["ceiling_light"]["target_count"], 1)

    def test_llm_contract_groups_are_sanitized_against_contract_objects(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": "desk",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": "a compact desk",
                        "reason": "explicitly requested",
                        "confidence": 0.9,
                    },
                    {
                        "object_type": "chair",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": "with a chair",
                        "reason": "explicitly requested",
                        "confidence": 0.9,
                    },
                    {
                        "object_type": "ottoman",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": "ottoman",
                        "reason": "not in brief",
                        "confidence": 0.4,
                    },
                ],
                "groups": [
                    {
                        "group_id": "workstation",
                        "members": ["desk", "chair", "ottoman"],
                        "intent": "functional_pairing",
                        "priority": "high",
                        "drop_policy": "preserve_core_first",
                    }
                ],
            },
            brief_text="Include a compact desk with a chair and a bookshelf.",
            available_object_types=["desk", "chair", "bookshelf"],
        )

        objects = {row["object_type"] for row in contract["objects"]}
        self.assertEqual(objects, {"desk", "chair"})
        self.assertEqual(contract["groups"][0]["members"], ["desk", "chair"])


if __name__ == "__main__":
    unittest.main()
