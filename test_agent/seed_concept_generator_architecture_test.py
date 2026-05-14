from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import agent.seed_concept_generator as seed_mod
from agent.seed_concept_generator import SeedConceptGenerator, solver_plan_from_concept


class SeedConceptGeneratorArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_seed_env = os.environ.get("TKNT_SEED_CONCEPT_GUIDED_LLM")
        self._previous_semantic_env = os.environ.get("TKNT_SEMANTIC_ADAPTIVE_LLM")
        os.environ["TKNT_SEED_CONCEPT_GUIDED_LLM"] = "0"
        os.environ["TKNT_SEMANTIC_ADAPTIVE_LLM"] = "0"

    def tearDown(self) -> None:
        if self._previous_seed_env is None:
            os.environ.pop("TKNT_SEED_CONCEPT_GUIDED_LLM", None)
        else:
            os.environ["TKNT_SEED_CONCEPT_GUIDED_LLM"] = self._previous_seed_env
        if self._previous_semantic_env is None:
            os.environ.pop("TKNT_SEMANTIC_ADAPTIVE_LLM", None)
        else:
            os.environ["TKNT_SEMANTIC_ADAPTIVE_LLM"] = self._previous_semantic_env

    def test_llm_guidance_can_refine_macro_concept_bundle(self) -> None:
        room_model = _room_model()
        clusters = _clusters()

        class _GuidanceClient:
            def chat_completion(self, *_args, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"family_order":["zoned","open_center","focal_axis","daylight_oriented","edge_weighted"],'
                                    '"concept_blueprints":[{"concept_family":"zoned",'
                                    '"topology_policy":{"reserve_center_degree":"medium","wall_loading_bias":"zone_specific"},'
                                    '"cluster_zone_overrides":[{"cluster_id":"main_seating","zone_assignment":"floating_center_zone","preferred_wall_side":"center","center_usage":"primary"},'
                                    '{"cluster_id":"media","zone_assignment":"right_wall_zone","preferred_wall_side":"right_wall","center_usage":"none"}],'
                                    '"notes":["Guided zoned concept."]}],'
                                    '"notes":["Macro concept guidance applied."]}'
                                )
                            )
                        )
                    ]
                )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "TKNT_SEED_CONCEPT_GUIDED_LLM": "1",
                    "TKNT_SEMANTIC_ADAPTIVE_LLM": "0",
                },
                clear=False,
            ),
            mock.patch.object(
                seed_mod, "get_llm_client", return_value=_GuidanceClient()
            ),
        ):
            bundle = SeedConceptGenerator().generate_bundle(
                room_model_json=room_model,
                clusters_json=clusters,
                target_count=5,
                description="Keep the room airy and socially legible.",
            )

        self.assertEqual(bundle["controller"], "llm_guided_seed_concept_generator")
        self.assertTrue(bundle["settings"]["llm_guided"])
        self.assertEqual(bundle["concepts"][0]["concept_family"], "zoned")
        zoned_concept = bundle["concepts"][0]
        main_row = next(
            row
            for row in zoned_concept["cluster_zone_plan"]
            if row["cluster_id"] == "main_seating"
        )
        media_row = next(
            row
            for row in zoned_concept["cluster_zone_plan"]
            if row["cluster_id"] == "media"
        )
        self.assertEqual(main_row["zone_assignment"], "floating_center_zone")
        self.assertEqual(main_row["center_usage"], "primary")
        self.assertEqual(media_row["zone_assignment"], "right_wall_zone")
        self.assertTrue(
            any(
                "LLM-guided SeedConceptGenerator refined macro concept blueprints"
                in note
                for note in bundle["notes"]
            )
        )

    def test_llm_guidance_falls_back_to_deterministic_bundle(self) -> None:
        room_model = _room_model()
        clusters = _clusters()

        class _InvalidGuidanceClient:
            def chat_completion(self, *_args, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="not valid json")
                        )
                    ]
                )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "TKNT_SEED_CONCEPT_GUIDED_LLM": "1",
                    "TKNT_SEMANTIC_ADAPTIVE_LLM": "0",
                },
                clear=False,
            ),
            mock.patch.object(
                seed_mod,
                "get_llm_client",
                return_value=_InvalidGuidanceClient(),
            ),
        ):
            bundle = SeedConceptGenerator().generate_bundle(
                room_model_json=room_model,
                clusters_json=clusters,
                target_count=5,
            )

        self.assertEqual(bundle["controller"], "deterministic_seed_concept_generator")
        self.assertFalse(bundle["settings"]["llm_guided"])
        self.assertTrue(
            any(
                "fell back to deterministic macro concept generation" in note
                for note in bundle["notes"]
            )
        )

    def test_concepts_emit_operational_solver_guidance(self) -> None:
        room_model = _room_model()
        clusters = _clusters()

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=5,
        )

        operational_zones = bundle["macro_region_map"]["protected_topology"][
            "operational_zones"
        ]
        self.assertTrue(operational_zones)
        self.assertTrue(
            all(
                {
                    "region",
                    "priority",
                    "enforcement",
                    "max_overlap_ratio",
                    "applies_to",
                    "violation_severity",
                }
                <= set(zone)
                for zone in operational_zones
            )
        )

        concepts = bundle["concepts"]
        self.assertEqual(len(concepts), 5)
        scenario_ids = {
            concept["macro_scenario"]["scenario_id"] for concept in concepts
        }
        self.assertEqual(len(scenario_ids), 5)
        for concept in concepts:
            self.assertIn("macro_scenario", concept)
            self.assertIn("anchor_region_preferences_by_cluster", concept)
            self.assertIn("anchor_pair_contracts", concept)
            allowed = concept["allowed_variant_families_by_cluster"]
            self.assertEqual(set(allowed), {"main_seating", "media", "storage"})
            self.assertTrue(
                all(2 <= len(families) <= 4 for families in allowed.values())
            )
            self.assertIn("conversation_facing", allowed["main_seating"])
            self.assertIn("media_facing", allowed["media"])

            core_rows = {
                row["cluster_id"]: row
                for row in concept["cluster_zone_plan"]
                if row["cluster_id"] in {"main_seating", "media"}
            }
            self.assertNotEqual(
                _zone_logic_class(core_rows["main_seating"]["zone_assignment"]),
                _zone_logic_class(core_rows["media"]["zone_assignment"]),
            )

            self.assertTrue(concept["critical_object_orientations"])
            self.assertTrue(
                all(
                    {"object_id", "intents", "priority"} <= set(item)
                    for item in concept["critical_object_orientations"]
                )
            )
            self.assertTrue(concept["primary_pair_contracts"])
            self.assertTrue(
                all(
                    {
                        "pair_type",
                        "cluster_a",
                        "cluster_b",
                        "strength",
                        "required",
                    }
                    <= set(contract)
                    for contract in concept["primary_pair_contracts"]
                )
            )
            self.assertIn(
                "required_publishable_diagnostics",
                concept["concept_readiness_requirements"],
            )
            self.assertIn("family_match_weight", concept["variant_bias_weights"])
            self.assertTrue(concept["macro_constraints"]["protected_topology"])
            anchor_hints = concept["anchor_region_preferences_by_cluster"]
            for cluster_id in ("main_seating", "media"):
                self.assertFalse(anchor_hints[cluster_id]["required_region_ids"])
                self.assertIn(
                    core_rows[cluster_id]["zone_assignment"],
                    anchor_hints[cluster_id]["preferred_region_ids"],
                )
                self.assertTrue(anchor_hints[cluster_id]["preferred_wall_side"])

        solver_plan = solver_plan_from_concept(
            concept=concepts[0],
            room_model_json=room_model,
            room_type="living_room",
        )
        self.assertTrue(solver_plan["object_orientations"])
        self.assertEqual(
            solver_plan["allowed_variant_families_by_cluster"],
            concepts[0]["allowed_variant_families_by_cluster"],
        )
        self.assertTrue(solver_plan["cluster_directional_relations"])
        self.assertTrue(solver_plan["anchor_pair_contracts"])
        self.assertTrue(
            solver_plan["anchor_layout_hints_by_cluster"]["main_seating"][
                "preferred_region_ids"
            ]
        )
        self.assertFalse(
            solver_plan["anchor_layout_hints_by_cluster"]["main_seating"][
                "required_region_ids"
            ]
        )

    def test_bedroom_concepts_emit_sleep_variant_guidance(self) -> None:
        room_model = _bedroom_room_model()
        clusters = _bedroom_clusters()

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=5,
        )

        concepts = bundle["concepts"]
        self.assertEqual(len(concepts), 5)

        for concept in concepts:
            allowed = concept["allowed_variant_families_by_cluster"]
            self.assertIn("sleep_area", allowed)
            sleep_families = allowed["sleep_area"]
            self.assertNotIn("base", sleep_families)
            self.assertNotIn("perimeter_facing", sleep_families)
            self.assertIn("headboard_wall_balanced", sleep_families)
            self.assertIn("bed_plus_storage_buffer", sleep_families)

            sleep_row = next(
                row
                for row in concept["cluster_zone_plan"]
                if row["cluster_id"] == "sleep_area"
            )
            self.assertEqual(sleep_row["role_kind"], "sleep")
            self.assertFalse(sleep_row["required_region_ids"])
            self.assertIn(
                sleep_row["zone_assignment"],
                sleep_row["preferred_region_ids"],
            )

        solver_plan = solver_plan_from_concept(
            concept=concepts[0],
            room_model_json=room_model,
            room_type="bedroom",
        )
        self.assertEqual(
            solver_plan["allowed_variant_families_by_cluster"],
            concepts[0]["allowed_variant_families_by_cluster"],
        )
        self.assertTrue(
            any(
                item.get("object_id") == "bed"
                for item in solver_plan["object_orientations"]
                if isinstance(item, dict)
            )
        )

    def test_living_profile_traits_do_not_relabel_seed_clusters_as_kitchen(
        self,
    ) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TKNT_ROOM_PROFILE_TRAITS_MODE": "canary",
                "TKNT_ROOM_PROFILE_TRAIT_ROOMS": "living_room",
            },
            clear=False,
        ):
            role_kind = seed_mod._role_kind(
                "main_seating",
                ["sofa", "coffee_table"],
                "",
            )

        self.assertEqual(role_kind, "social_anchor")


def _zone_logic_class(zone: object) -> str:
    text = str(zone or "")
    if "focal" in text:
        return "focal"
    if "daylight" in text:
        return "daylight"
    if "edge" in text or "storage" in text:
        return "perimeter"
    if "floating" in text or "center" in text:
        return "center"
    return text


def _room_model() -> dict[str, object]:
    return {
        "room": {
            "room_id": "living_test",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 5000, "y": 0},
                {"x": 5000, "y": 3600},
                {"x": 0, "y": 3600},
            ],
        },
        "meta": {"room_type": "living_room"},
        "openings": {
            "doors": [{"id": "entry_door"}],
            "windows": [{"id": "main_window"}],
        },
        "affordance_map": {
            "focal_surfaces": ["wall_north"],
            "usable_wall_segments": ["wall_north", "wall_south", "wall_east"],
            "entry_landing_zones": ["entry_landing"],
            "primary_circulation_corridors": ["entry_to_center"],
            "center_openness_regions": ["room_center"],
            "daylight_regions": ["main_window_zone"],
            "wall_anchor_candidates": ["wall_south", "wall_east"],
        },
    }


def _clusters() -> dict[str, object]:
    return {
        "clusters": [
            {
                "cluster_id": "main_seating",
                "members": ["sectional_sofa", "coffee_table"],
                "anchors": ["sectional_sofa"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "medium",
                        "daylight_affinity": "medium",
                        "floating_allowed": True,
                    }
                },
            },
            {
                "cluster_id": "media",
                "members": ["tv_console"],
                "anchors": ["tv_console"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "daylight_affinity": "low",
                        "floating_allowed": False,
                    }
                },
            },
            {
                "cluster_id": "storage",
                "members": ["bookcase"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "floating_allowed": False,
                    }
                },
            },
        ]
    }


def _bedroom_room_model() -> dict[str, object]:
    return {
        "room": {
            "room_id": "bedroom_test",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 5200, "y": 0},
                {"x": 5200, "y": 3900},
                {"x": 0, "y": 3900},
            ],
        },
        "meta": {"room_type": "bedroom"},
        "openings": {
            "doors": [{"id": "entry_door"}],
            "windows": [{"id": "bedroom_window"}],
        },
        "affordance_map": {
            "usable_wall_segments": ["wall_north", "wall_south", "wall_east"],
            "entry_landing_zones": ["entry_landing"],
            "primary_circulation_corridors": ["entry_to_bed"],
            "center_openness_regions": ["room_center"],
            "daylight_regions": ["window_zone"],
            "wall_anchor_candidates": ["wall_north", "wall_east"],
        },
    }


def _bedroom_clusters() -> dict[str, object]:
    return {
        "clusters": [
            {
                "cluster_id": "sleep_area",
                "members": ["bed", "nightstand", "dresser"],
                "anchors": ["bed"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "daylight_affinity": "medium",
                        "floating_allowed": False,
                    }
                },
            },
            {
                "cluster_id": "work_area",
                "members": ["desk", "office_chair"],
                "anchors": ["desk"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "medium",
                        "daylight_affinity": "high",
                        "floating_allowed": False,
                    }
                },
            },
            {
                "cluster_id": "storage_area",
                "members": ["wardrobe"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "daylight_affinity": "low",
                        "floating_allowed": False,
                    }
                },
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
