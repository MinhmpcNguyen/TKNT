from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_request_contract = importlib.import_module("agent.request_contract")
build_request_contract = _request_contract.build_request_contract
contract_item_for_object_type = _request_contract.contract_item_for_object_type
missing_non_functional_contract_items = (
    _request_contract.missing_non_functional_contract_items
)
sanitize_request_contract = _request_contract.sanitize_request_contract


class RequestContractTest(unittest.TestCase):
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
