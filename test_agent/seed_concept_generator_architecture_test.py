from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import agent.seed_concept_generator as seed_mod
from agent.seed_concept_generator import SeedConceptGenerator, solver_plan_from_concept


class SeedConceptGeneratorArchitectureTest(unittest.TestCase):
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

        with mock.patch.object(
            seed_mod, "get_llm_client", return_value=_GuidanceClient()
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

    def test_llm_guidance_failure_raises_without_deterministic_fallback(self) -> None:
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
            mock.patch.object(
                seed_mod,
                "get_llm_client",
                return_value=_InvalidGuidanceClient(),
            ),
            self.assertRaises(ValueError),
        ):
            SeedConceptGenerator().generate_bundle(
                room_model_json=room_model,
                clusters_json=clusters,
                target_count=5,
            )

    def test_llm_guidance_rejects_duplicate_projected_physical_signatures(
        self,
    ) -> None:
        room_model = _room_model()
        clusters = _clusters()

        class _DuplicateGuidanceClient:
            def chat_completion(self, *_args, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"family_order":["zoned","daylight_oriented"],'
                                    '"concept_blueprints":['
                                    '{"concept_family":"zoned","intent_label":"social center",'
                                    '"primary_cluster_region":{"zone_assignment":"floating_center_zone","preferred_wall_side":"center","center_usage":"primary"},'
                                    '"secondary_cluster_region":{"zone_assignment":"right_wall_zone","preferred_wall_side":"right_wall","center_usage":"none"},'
                                    '"support_region":{"zone_assignment":"left_wall_zone","preferred_wall_side":"left_wall","center_usage":"none"},'
                                    '"preferred_wall_side":"","center_usage":"none","topology_policy":{"wall_loading_bias":"zone_specific"}},'
                                    '{"concept_family":"daylight_oriented","intent_label":"renamed duplicate",'
                                    '"primary_cluster_region":{"zone_assignment":"floating_center_zone","preferred_wall_side":"center","center_usage":"primary"},'
                                    '"secondary_cluster_region":{"zone_assignment":"right_wall_zone","preferred_wall_side":"right_wall","center_usage":"none"},'
                                    '"support_region":{"zone_assignment":"left_wall_zone","preferred_wall_side":"left_wall","center_usage":"none"},'
                                    '"preferred_wall_side":"","center_usage":"none","topology_policy":{"wall_loading_bias":"balanced"}}],'
                                    '"notes":[]}'
                                )
                            )
                        )
                    ]
                )

        with mock.patch.object(
            seed_mod, "get_llm_client", return_value=_DuplicateGuidanceClient()
        ):
            bundle = SeedConceptGenerator().generate_bundle(
                room_model_json=room_model,
                clusters_json=clusters,
                target_count=2,
                description="Keep physical options genuinely distinct.",
            )

        validation = bundle["llm_guidance_validation"]
        self.assertEqual(validation["accepted_concept_count"], 1)
        self.assertEqual(validation["rejected_concept_count"], 1)
        self.assertEqual(
            validation["rejected_concepts"][0]["reason"],
            "duplicate_projected_physical_signature",
        )
        self.assertEqual(len(bundle["concepts"]), 1)

    def test_invalid_guided_topology_values_are_not_silently_accepted(self) -> None:
        room_model = _room_model()
        clusters = _clusters()

        class _InvalidTopologyGuidanceClient:
            def chat_completion(self, *_args, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"family_order":["zoned"],'
                                    '"concept_blueprints":[{"concept_family":"zoned",'
                                    '"intent_label":"invalid topology",'
                                    '"primary_cluster_region":"floating_center_zone",'
                                    '"secondary_cluster_region":"right_wall_zone",'
                                    '"support_region":"left_wall_zone",'
                                    '"preferred_wall_side":"right_wall",'
                                    '"center_usage":"primary",'
                                    '"topology_policy":{"wall_loading_bias":"segmented",'
                                    '"entry_avoidance_strength":"opposite_entry",'
                                    '"daylight_bias":"daylight_side"}}],'
                                    '"notes":[]}'
                                )
                            )
                        )
                    ]
                )

        with mock.patch.object(
            seed_mod, "get_llm_client", return_value=_InvalidTopologyGuidanceClient()
        ):
            bundle = SeedConceptGenerator().generate_bundle(
                room_model_json=room_model,
                clusters_json=clusters,
                target_count=1,
            )

        sanitized_policy = bundle["llm_guidance"]["concept_blueprints"][0][
            "topology_policy"
        ]
        self.assertEqual(sanitized_policy, {})
        concept_policy = bundle["concepts"][0]["topology_policy"]
        self.assertNotIn("segmented", set(concept_policy.values()))
        self.assertNotIn("opposite_entry", set(concept_policy.values()))
        self.assertNotIn("daylight_side", set(concept_policy.values()))

    def test_concepts_emit_operational_solver_guidance(self) -> None:
        room_model = _room_model()
        clusters = _clusters()

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=5,
            use_llm=False,
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
        planned_material_signatures = {
            concept["planned_material_macro_signature"] for concept in concepts
        }
        self.assertGreaterEqual(len(planned_material_signatures), 3)
        for concept in concepts:
            self.assertIn("macro_scenario", concept)
            self.assertIn("material_macro_signature_target", concept)
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
                self.assertEqual(
                    anchor_hints[cluster_id]["committed_region_ids"],
                    [core_rows[cluster_id]["zone_assignment"]],
                )
                self.assertEqual(
                    anchor_hints[cluster_id]["committed_wall_side"],
                    core_rows[cluster_id]["committed_wall_side"],
                )
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
        first_main_row = next(
            row
            for row in concepts[0]["cluster_zone_plan"]
            if row["cluster_id"] == "main_seating"
        )
        self.assertEqual(
            solver_plan["anchor_layout_hints_by_cluster"]["main_seating"][
                "committed_region_ids"
            ],
            first_main_row["committed_region_ids"],
        )
        self.assertFalse(
            solver_plan["anchor_layout_hints_by_cluster"]["main_seating"][
                "required_region_ids"
            ]
        )

    def test_entry_support_shoe_rack_prefers_entry_buffer_zone(self) -> None:
        room_model = _room_model()
        clusters = _clusters()
        cluster_rows = clusters["clusters"]
        assert isinstance(cluster_rows, list)
        cluster_rows.append(
            {
                "cluster_id": "pet_or_entry_support",
                "members": ["shoe_rack"],
                "anchors": ["shoe_rack"],
                "cluster_rules": {
                    "zone_claims": {
                        "preferred_regions": ["wall_south", "wall_east"],
                        "avoid_regions": ["entry_landing", "entry_to_center"],
                        "wall_affinity": "high",
                        "daylight_affinity": "medium",
                        "floating_allowed": False,
                    }
                },
            }
        )

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=1,
            use_llm=False,
        )

        concept = bundle["concepts"][0]
        shoe_row = next(
            row
            for row in concept["cluster_zone_plan"]
            if row["cluster_id"] == "pet_or_entry_support"
        )
        self.assertEqual(shoe_row["zone_assignment"], "entry_adjacent_active_zone")
        self.assertEqual(shoe_row["entry_relation"], "buffer_entry_if_possible")
        self.assertIn("entry_landing", shoe_row["forbidden_region_ids"])
        self.assertIn("entry_to_center", shoe_row["forbidden_region_ids"])

        plan = solver_plan_from_concept(concept=concept, room_model_json=room_model)
        affinity = next(
            row
            for row in plan["cluster_affinities"]
            if row["cluster_id"] == "pet_or_entry_support"
        )
        self.assertIn("entry_side", affinity["prefer"])
        self.assertIn("near_entry", affinity["prefer"])
        self.assertNotIn("far_from_entry", affinity["prefer"])

    def test_bedroom_concepts_emit_sleep_variant_guidance(self) -> None:
        room_model = _bedroom_room_model()
        clusters = _bedroom_clusters()

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=5,
            use_llm=False,
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
            self.assertEqual(
                sleep_row["committed_region_ids"],
                [sleep_row["zone_assignment"]],
            )
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

    def test_request_contract_viewing_pair_projects_to_object_and_cluster_contracts(
        self,
    ) -> None:
        room_model = _bedroom_room_model()
        clusters = _bedroom_media_clusters()
        request_contract = clusters["request_contract"]
        assert isinstance(request_contract, dict)
        objects = request_contract["objects"]
        assert isinstance(objects, list)
        objects[0] = {
            "object_type": "bed",
            "intent": "must_keep",
            "evidence": "bed faces the tv console",
            "relation_edges": [{"relation": "face", "target": "tv_console"}],
        }

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=1,
            use_llm=False,
        )

        concept = bundle["concepts"][0]
        self.assertEqual(
            concept["object_viewing_pairs"],
            [
                {
                    "type": "viewing_pair",
                    "viewer_cluster_id": "sleep_core",
                    "viewer_object_id": "bed",
                    "viewer_object_type": "bed",
                    "target_cluster_id": "media_optional",
                    "target_object_id": "tv_console",
                    "target_object_type": "tv_console",
                    "relation": "view_target",
                    "required": True,
                    "priority": "high",
                    "source": "request_contract_relation_edge",
                    "evidence": "bed faces the tv console",
                }
            ],
        )
        self.assertEqual(
            concept["primary_pair_contracts"][0]["pair_type"],
            "view_target",
        )
        self.assertEqual(
            concept["primary_pair_contracts"][0]["source"],
            "object_viewing_pair",
        )

        solver_plan = solver_plan_from_concept(
            concept=concept,
            room_model_json=room_model,
            room_type="bedroom",
        )

        self.assertEqual(
            solver_plan["object_viewing_pairs"],
            concept["object_viewing_pairs"],
        )
        self.assertEqual(
            solver_plan["cluster_directional_relations"],
            [
                {
                    "a": "sleep_core",
                    "b": "media_optional",
                    "relation": "view_target",
                    "priority": "high",
                    "reason": "Projection from primary_pair_contract=view_target.",
                }
            ],
        )

    def test_bed_and_tv_without_explicit_relation_does_not_create_view_target(
        self,
    ) -> None:
        room_model = _bedroom_room_model()
        clusters = _bedroom_media_clusters()
        request_contract = clusters["request_contract"]
        assert isinstance(request_contract, dict)
        request_contract["groups"] = []

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=1,
            use_llm=False,
        )

        self.assertEqual(bundle["concepts"][0]["object_viewing_pairs"], [])

    def test_support_layout_role_does_not_block_bedroom_primary_secondary_inference(
        self,
    ) -> None:
        room_model = _bedroom_room_model()
        clusters = _bedroom_media_clusters()
        request_contract = clusters["request_contract"]
        assert isinstance(request_contract, dict)
        objects = request_contract["objects"]
        assert isinstance(objects, list)
        objects[0] = {
            "object_type": "bed",
            "intent": "must_keep",
            "relation_edges": [{"relation": "face", "target": "tv_console"}],
        }
        clusters["semantic_layout_program"] = {
            "active_clusters": [
                {
                    "cluster_id": "sleep_core",
                    "layout_role": "support",
                    "priority": "core",
                    "semantic_role": "primary_anchor_zone",
                },
                {
                    "cluster_id": "media_optional",
                    "layout_role": "support",
                    "priority": "support",
                    "semantic_role": "secondary_support_zone",
                },
                {
                    "cluster_id": "storage_core",
                    "layout_role": "support",
                    "priority": "core",
                    "semantic_role": "storage_zone",
                },
            ]
        }

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=1,
            use_llm=False,
        )

        rows = {
            row["cluster_id"]: row for row in bundle["concepts"][0]["cluster_zone_plan"]
        }
        self.assertEqual(rows["sleep_core"]["scenario_role"], "primary")
        self.assertEqual(rows["media_optional"]["scenario_role"], "secondary")
        self.assertEqual(rows["storage_core"]["scenario_role"], "support")
        self.assertEqual(
            bundle["concepts"][0]["primary_pair_contracts"][0]["pair_type"],
            "view_target",
        )

    def test_request_contract_relation_edge_can_create_viewing_pair(self) -> None:
        room_model = _bedroom_room_model()
        clusters = _bedroom_media_clusters()
        request_contract = clusters["request_contract"]
        assert isinstance(request_contract, dict)
        request_contract["groups"] = []
        objects = request_contract["objects"]
        assert isinstance(objects, list)
        objects[1] = {
            "object_type": "tv_console",
            "intent": "must_keep",
            "evidence": "A tv should face the bed",
            "reason": "TV support is required to satisfy the facing relationship.",
            "relation_edges": [{"relation": "faces", "target": "bed"}],
        }

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=1,
            use_llm=False,
        )

        pair = bundle["concepts"][0]["object_viewing_pairs"][0]
        self.assertEqual(pair["viewer_object_id"], "bed")
        self.assertEqual(pair["target_object_id"], "tv_console")
        self.assertEqual(pair["relation"], "view_target")
        self.assertEqual(pair["source"], "request_contract_relation_edge")

    def test_prose_evidence_does_not_create_passive_support_viewing_pair(
        self,
    ) -> None:
        room_model = _room_model()
        clusters = _clusters()
        cluster_rows = clusters["clusters"]
        assert isinstance(cluster_rows, list)
        main_seating = cluster_rows[0]
        assert isinstance(main_seating, dict)
        main_seating["members"] = ["sofa", "coffee_table"]
        main_seating["anchors"] = ["sofa"]
        clusters["request_contract"] = {
            "objects": [
                {
                    "object_type": "sofa",
                    "intent": "must_keep",
                    "evidence": "sofa facing a tv console",
                    "relation_edges": [{"relation": "facing", "target": "tv_console"}],
                },
                {
                    "object_type": "coffee_table",
                    "intent": "must_keep",
                    "evidence": "include a sofa facing a TV console, a coffee table, a rug",
                },
                {"object_type": "tv_console", "intent": "must_keep"},
            ],
            "groups": [],
        }

        bundle = SeedConceptGenerator().generate_bundle(
            room_model_json=room_model,
            clusters_json=clusters,
            target_count=1,
            use_llm=False,
        )

        pairs = bundle["concepts"][0]["object_viewing_pairs"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["viewer_object_id"], "sofa")
        self.assertEqual(pairs[0]["target_object_id"], "tv_console")
        self.assertNotIn(
            "coffee_table",
            {str(pair.get("viewer_object_id")) for pair in pairs},
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


def _bedroom_media_clusters() -> dict[str, object]:
    return {
        "request_contract": {
            "objects": [
                {"object_type": "bed", "intent": "must_keep"},
                {"object_type": "tv_console", "intent": "must_keep"},
                {"object_type": "wardrobe", "intent": "must_keep"},
            ],
            "groups": [
                {
                    "group_id": "media_pair",
                    "members": ["bed", "tv_console"],
                    "intent": "composition",
                    "priority": "core",
                    "drop_policy": "tv must remain oriented to bed",
                }
            ],
        },
        "clusters": [
            {
                "cluster_id": "sleep_core",
                "members": ["bed", "nightstand"],
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
                "cluster_id": "media_optional",
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
                "cluster_id": "storage_core",
                "members": ["wardrobe"],
                "anchors": ["wardrobe"],
                "cluster_rules": {
                    "zone_claims": {
                        "wall_affinity": "high",
                        "daylight_affinity": "low",
                        "floating_allowed": False,
                    }
                },
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
