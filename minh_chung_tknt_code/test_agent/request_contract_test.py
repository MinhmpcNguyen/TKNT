from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

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


def _llm_contract_object(
    object_type: str,
    evidence: str,
    *,
    placement_participation: str = "floor_blocking",
) -> dict[str, object]:
    return {
        "object_type": object_type,
        "intent": "must_keep",
        "min_keep": 1,
        "target_count": 1,
        "max_keep": 1,
        "evidence": evidence,
        "reason": "explicitly requested",
        "confidence": 0.9,
        "placement_participation": placement_participation,
    }


def _members_by_cluster(program: dict[str, object]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for cluster in program.get("active_clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "")
        members: set[str] = set()
        for bundle in cluster.get("required_bundles") or []:
            if not isinstance(bundle, dict):
                continue
            for item in bundle.get("objects") or []:
                if isinstance(item, dict) and item.get("object_type"):
                    members.add(str(item["object_type"]))
        out[cluster_id] = members
    return out


class RequestContractTest(unittest.TestCase):
    def test_attach_keeps_llm_empty_contract(
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

        result = attach_request_contract_to_semantic_program(
            semantic_program,
            brief_text="Include a desk.",
        )

        contract = result["request_contract"]
        self.assertEqual(contract["source"], "llm_request_contract")
        self.assertEqual(contract["objects"], [])

    def test_attach_raises_when_llm_contract_is_missing(
        self,
    ) -> None:
        semantic_program = {
            "active_clusters": [{"cluster_id": "work", "members": ["desk"]}]
        }

        with self.assertRaises(ValueError):
            attach_request_contract_to_semantic_program(
                semantic_program,
                brief_text="Include a desk.",
            )

    def test_sanitizer_separates_inventory_from_program_availability(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    _llm_contract_object("chair", "a chair"),
                ],
                "groups": [],
            },
            brief_text="Include a chair.",
            available_object_types=["bed"],
            inventory_object_types=["bed", "chair"],
        )

        item = contract["objects"][0]
        self.assertTrue(item["available_in_inventory"])
        self.assertTrue(item["inventory_available"])
        self.assertFalse(item["available_in_program"])
        self.assertEqual(item["solver_participation"], "hard_solver_object")

    def test_attach_reconciles_requested_bedroom_pairs_generically(self) -> None:
        brief = (
            "1 bed, 1 wardrobe, 1 table next to bed with a lamp, "
            "1 bean bag, 1 chair in front of table, 1 big round rug on the floor"
        )
        semantic_program = {
            "status": "OK",
            "room_type": "bedroom",
            "active_clusters": [
                {
                    "cluster_id": "sleep_core",
                    "priority": "core",
                    "layout_role": "primary",
                    "semantic_role": "sleep",
                    "dominant_anchor_candidates": ["bed"],
                    "required_bundles": [
                        {
                            "bundle_id": "sleep_core_bundle",
                            "objects": [
                                {"object_type": "bed", "role": "dominant_anchor"},
                                {"object_type": "nightstand", "role": "support"},
                            ],
                        }
                    ],
                },
                {
                    "cluster_id": "storage_core",
                    "priority": "core",
                    "layout_role": "support",
                    "semantic_role": "storage",
                    "required_bundles": [
                        {
                            "bundle_id": "storage_core_bundle",
                            "objects": [{"object_type": "wardrobe", "role": "support"}],
                        }
                    ],
                },
            ],
            "request_contract": {
                "source": "llm_request_contract",
                "objects": [
                    _llm_contract_object("bed", "1 bed"),
                    _llm_contract_object("wardrobe", "1 wardrobe"),
                    _llm_contract_object("side_table", "1 table next to bed"),
                    _llm_contract_object("floor_lamp", "with a lamp"),
                    _llm_contract_object("armchair", "1 bean bag"),
                    _llm_contract_object("chair", "1 chair in front of table"),
                    _llm_contract_object(
                        "rug",
                        "1 big round rug on the floor",
                        placement_participation="textile_overlay",
                    ),
                ],
                "groups": [],
            },
        }

        result = attach_request_contract_to_semantic_program(
            semantic_program,
            brief_text=brief,
            inventory_object_types=[
                "bed",
                "wardrobe",
                "nightstand",
                "desk",
                "chair",
                "bean_bag",
                "floor_lamp",
                "rug",
            ],
        )

        members_by_cluster = _members_by_cluster(result)
        self.assertIn("desk", members_by_cluster["work_study"])
        self.assertIn("chair", members_by_cluster["work_study"])
        self.assertIn("bean_bag", members_by_cluster["lounge_reading"])
        contract_objects = {
            row["object_type"]: row for row in result["request_contract"]["objects"]
        }
        self.assertEqual(contract_objects["side_table"]["fulfilled_by"], "nightstand")
        self.assertEqual(
            contract_objects["desk"]["request_reconciliation_status"], "reconciled"
        )
        self.assertEqual(contract_objects["armchair"]["chosen_object_type"], "bean_bag")
        self.assertEqual(
            contract_objects["armchair"]["request_reconciliation_status"],
            "substituted",
        )
        self.assertEqual(
            contract_objects["rug"]["solver_participation"],
            "post_solver_accessory",
        )

    def test_attach_reconciles_living_room_requested_seating_and_lighting(
        self,
    ) -> None:
        brief = (
            "Living room with a sofa, bean bag, coffee table, side table, "
            "and floor lamp."
        )
        semantic_program = {
            "status": "OK",
            "room_type": "living_room",
            "active_clusters": [
                {
                    "cluster_id": "main_seating",
                    "priority": "core",
                    "layout_role": "primary",
                    "semantic_role": "living",
                    "dominant_anchor_candidates": ["sofa"],
                    "required_bundles": [
                        {
                            "bundle_id": "main_seating_bundle",
                            "objects": [
                                {"object_type": "sofa", "role": "dominant_anchor"},
                                {"object_type": "coffee_table", "role": "support"},
                            ],
                        }
                    ],
                }
            ],
            "request_contract": {
                "source": "llm_request_contract",
                "objects": [
                    _llm_contract_object("sofa", "a sofa"),
                    _llm_contract_object("bean_bag", "bean bag"),
                    _llm_contract_object("coffee_table", "coffee table"),
                    _llm_contract_object("side_table", "side table"),
                    _llm_contract_object("floor_lamp", "floor lamp"),
                ],
                "groups": [],
            },
        }

        result = attach_request_contract_to_semantic_program(
            semantic_program,
            brief_text=brief,
            inventory_object_types=[
                "sofa",
                "bean_bag",
                "coffee_table",
                "side_table",
                "floor_lamp",
            ],
        )

        members = _members_by_cluster(result)["main_seating"]
        self.assertIn("bean_bag", members)
        self.assertIn("side_table", members)
        self.assertIn("floor_lamp", members)

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
            brief_text="Include a bed, a rug, a mirror, and a ceiling lamp.",
            available_object_types=["bed"],
        )

        missing = missing_non_functional_contract_items(
            contract=contract,
            objects=[{"object_type": "bed"}],
        )

        self.assertEqual(
            [row["object_type"] for row in missing],
            ["ceiling_light", "mirror", "rug"],
        )
        self.assertTrue(
            all(
                row["solver_participation"] == "post_solver_accessory"
                for row in missing
            )
        )

    def test_plain_mirror_is_wall_attached_post_solver_accessory(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": "mirror",
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": "a mirror",
                        "reason": "explicitly requested",
                        "confidence": 0.9,
                        "placement_participation": "floor_blocking",
                    }
                ],
                "groups": [],
            },
            brief_text="Include a mirror.",
            available_object_types=["mirror"],
        )

        item = contract["objects"][0]
        self.assertEqual(item["placement_participation"], "wall_attached")
        self.assertEqual(item["solver_participation"], "post_solver_accessory")
        self.assertFalse(item["blocks_floor_area"])
        self.assertTrue(item["can_overlap_floor_objects"])

    def test_standing_mirror_can_enter_solver(self) -> None:
        contract = build_request_contract(
            brief_text="Include a standing mirror.",
            available_object_types=["mirror"],
        )

        item = contract["objects"][0]
        self.assertEqual(item["placement_participation"], "floor_blocking")
        self.assertEqual(item["solver_participation"], "hard_solver_object")
        self.assertTrue(item["blocks_floor_area"])

    def test_llm_floor_solver_guardrail_covers_bedroom_and_living_room(
        self,
    ) -> None:
        cases = [
            ("tv_console", "with a tv console", "wall_attached", "hard_solver_object"),
            ("bookshelf", "with a bookshelf", "wall_attached", "hard_solver_object"),
            ("bed", "with a bed", "decor_only", "hard_solver_object"),
            ("nightstand", "with a nightstand", "wall_attached", "hard_solver_object"),
            ("wardrobe", "with a wardrobe", "wall_attached", "hard_solver_object"),
            ("dresser", "with a dresser", "wall_attached", "hard_solver_object"),
            ("desk", "with a desk", "wall_attached", "hard_solver_object"),
            ("chair", "with a chair", "wall_attached", "hard_solver_object"),
            ("sofa", "with a sofa", "decor_only", "hard_solver_object"),
            ("armchair", "with an armchair", "decor_only", "hard_solver_object"),
            (
                "coffee_table",
                "with a coffee table",
                "wall_attached",
                "hard_solver_object",
            ),
            (
                "media_shelf",
                "with a media shelf",
                "wall_attached",
                "hard_solver_object",
            ),
            ("side_table", "with a side table", "wall_attached", "hard_solver_object"),
            ("floor_lamp", "with a floor lamp", "decor_only", "hard_solver_object"),
            (
                "storage_cabinet",
                "with a storage cabinet",
                "wall_attached",
                "hard_solver_object",
            ),
            (
                "console_table",
                "with a console table",
                "wall_attached",
                "hard_solver_object",
            ),
            ("ottoman", "with an ottoman", "decor_only", "hard_solver_object"),
        ]
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": object_type,
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": evidence,
                        "reason": "explicitly requested",
                        "confidence": 0.9,
                        "placement_participation": raw_placement,
                    }
                    for object_type, evidence, raw_placement, _ in cases
                ],
                "groups": [],
            },
            brief_text=", ".join(evidence for _, evidence, _, _ in cases),
            available_object_types=[object_type for object_type, _, _, _ in cases],
        )

        objects = {row["object_type"]: row for row in contract["objects"]}
        for object_type, _, raw_placement, solver_participation in cases:
            with self.subTest(object_type=object_type):
                item = objects[object_type]
                self.assertEqual(item["placement_participation"], "floor_blocking")
                self.assertEqual(item["solver_participation"], solver_participation)
                self.assertTrue(item["blocks_floor_area"])
                self.assertFalse(item["can_overlap_floor_objects"])
                self.assertTrue(item["placement_guardrail_applied"])
                self.assertEqual(item["raw_placement_participation"], raw_placement)

    def test_post_solver_decor_contract_types_ignore_llm_floor_override(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": object_type,
                        "intent": "must_keep",
                        "min_keep": 1,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": evidence,
                        "reason": "explicitly requested",
                        "confidence": 0.9,
                        "placement_participation": "floor_blocking",
                    }
                    for object_type, evidence in (
                        ("rug", "a rug"),
                        ("wall_art", "wall art"),
                        ("curtain", "curtains"),
                    )
                ],
                "groups": [],
            },
            brief_text="Include a rug, wall art, and curtains.",
            available_object_types=["rug", "wall_art", "curtain"],
        )

        objects = {row["object_type"]: row for row in contract["objects"]}
        self.assertEqual(objects["rug"]["placement_participation"], "textile_overlay")
        self.assertEqual(
            objects["wall_art"]["placement_participation"], "wall_attached"
        )
        self.assertEqual(
            objects["curtain"]["placement_participation"], "opening_attached"
        )
        self.assertTrue(
            all(
                row["solver_participation"] == "post_solver_accessory"
                for row in objects.values()
            )
        )

    def test_floor_solver_guardrail_preserves_soft_request_intent(self) -> None:
        contract = sanitize_request_contract(
            {
                "source": "llm_request_contract",
                "objects": [
                    {
                        "object_type": "armchair",
                        "intent": "target_if_viable",
                        "min_keep": 0,
                        "target_count": 1,
                        "max_keep": 1,
                        "evidence": "an armchair if possible",
                        "reason": "fit dependent",
                        "confidence": 0.9,
                        "placement_participation": "wall_attached",
                    }
                ],
                "groups": [],
            },
            brief_text="Add an armchair if possible.",
            available_object_types=["armchair"],
        )

        item = contract["objects"][0]
        self.assertEqual(item["intent"], "target_if_viable")
        self.assertEqual(item["placement_participation"], "floor_blocking")
        self.assertEqual(item["solver_participation"], "soft_solver_object")
        self.assertTrue(item["placement_guardrail_applied"])

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
