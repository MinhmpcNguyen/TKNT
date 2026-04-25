from __future__ import annotations

import unittest

from agent.seed_concept_generator import SeedConceptGenerator, solver_plan_from_concept


class SeedConceptGeneratorArchitectureTest(unittest.TestCase):
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
                self.assertTrue(anchor_hints[cluster_id]["required_region_ids"])
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
