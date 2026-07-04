from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

snapshot_prompt_compiler = importlib.import_module("pipeline.snapshot_prompt_compiler")


def _visible_object(
    *,
    object_id: str,
    object_type: str,
    distance_to_camera_m: float,
    screen_bbox_px: dict[str, float],
    color_hex: str,
    material: str,
    width_mm: float,
    depth_mm: float,
    height_mm: float,
    screen_center_px: dict[str, float] | None = None,
    place_on: dict[str, object] | None = None,
    plan_position_mm: dict[str, float] | None = None,
    plan_rotation_deg: float = 0,
) -> dict[str, object]:
    center = screen_center_px or {
        "x": (screen_bbox_px["minX"] + screen_bbox_px["maxX"]) / 2.0,
        "y": (screen_bbox_px["minY"] + screen_bbox_px["maxY"]) / 2.0,
    }
    return {
        "id": object_id,
        "type": object_type,
        "colorHex": color_hex,
        "material": material,
        "bboxMm": {
            "minX": 100,
            "minY": 100,
            "maxX": 100 + width_mm,
            "maxY": 100 + depth_mm,
        },
        "planPositionMm": plan_position_mm
        or {"x": 1000, "y": 1000, "z": height_mm / 2.0},
        "dimensionsMm": {
            "width": width_mm,
            "depth": depth_mm,
            "height": height_mm,
        },
        "planRotationDeg": plan_rotation_deg,
        "sceneYawDeg": 0,
        "placeOn": place_on,
        "distanceToCameraM": distance_to_camera_m,
        "screenCenterPx": center,
        "screenBboxPx": {
            **screen_bbox_px,
            "width": screen_bbox_px["maxX"] - screen_bbox_px["minX"],
            "height": screen_bbox_px["maxY"] - screen_bbox_px["minY"],
        },
        "visibleSampleCount": 4,
        "visibleSampleFraction": 0.444,
    }


def _snapshot(
    *,
    polygon_mm: list[dict[str, float]] | None = None,
    camera_position_room_mm: dict[str, float] | None = None,
    target_room_mm: dict[str, float] | None = None,
    visible_objects: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    objects = visible_objects or [
        _visible_object(
            object_id="tv_console",
            object_type="tv_console",
            distance_to_camera_m=4.2,
            screen_bbox_px={"minX": 500, "minY": 300, "maxX": 650, "maxY": 430},
            color_hex="#6F6962",
            material="wood",
            width_mm=1260,
            depth_mm=380,
            height_mm=590,
        ),
        _visible_object(
            object_id="sectional_sofa",
            object_type="sectional_sofa",
            distance_to_camera_m=5.9,
            screen_bbox_px={"minX": 470, "minY": 245, "maxX": 790, "maxY": 450},
            color_hex="#B9A690",
            material="fabric",
            width_mm=2540,
            depth_mm=920,
            height_mm=790,
        ),
        _visible_object(
            object_id="armchair_1",
            object_type="armchair",
            distance_to_camera_m=5.0,
            screen_bbox_px={"minX": 320, "minY": 280, "maxX": 420, "maxY": 405},
            color_hex="#D7C8B8",
            material="fabric",
            width_mm=840,
            depth_mm=710,
            height_mm=900,
        ),
        _visible_object(
            object_id="media_shelf_1",
            object_type="media_shelf",
            distance_to_camera_m=4.8,
            screen_bbox_px={"minX": 700, "minY": 285, "maxX": 785, "maxY": 390},
            color_hex="#CFCAC2",
            material="wood",
            width_mm=870,
            depth_mm=300,
            height_mm=800,
        ),
        _visible_object(
            object_id="vase_1",
            object_type="vase",
            distance_to_camera_m=4.4,
            screen_bbox_px={"minX": 560, "minY": 270, "maxX": 575, "maxY": 292},
            color_hex="#D7C8B8",
            material="ceramic",
            width_mm=240,
            depth_mm=220,
            height_mm=210,
            place_on={"method": "on_top", "targetId": "coffee_table"},
        ),
        _visible_object(
            object_id="wall_art_1",
            object_type="wall_art",
            distance_to_camera_m=6.8,
            screen_bbox_px={"minX": 760, "minY": 90, "maxX": 820, "maxY": 150},
            color_hex="#CFCAC2",
            material="wood",
            width_mm=420,
            depth_mm=35,
            height_mm=440,
            place_on={"method": "hang_on", "targetId": "wall"},
        ),
    ]
    return {
        "capturedAt": "2026-04-06T10:03:00.959Z",
        "variantId": "variant_1",
        "canvas": {"widthPx": 1345, "heightPx": 590},
        "room": {
            "boundsMm": {"minX": 0, "minY": 0, "maxX": 6000, "maxY": 4000},
            "polygonMm": polygon_mm
            or [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
            "surfaceColors": {
                "wallColorHex": "#F2ECE4",
                "floorColorHex": "#BC926A",
                "ceilingColorHex": "#FBF7F2",
            },
            "openings": {
                "doors": [
                    {
                        "id": "door_1",
                        "segmentMm": [
                            {"x": 5800, "y": 900},
                            {"x": 5800, "y": 1700},
                        ],
                    }
                ],
                "windows": [
                    {
                        "id": "window_1",
                        "segmentMm": [
                            {"x": 2100, "y": 0},
                            {"x": 3900, "y": 0},
                        ],
                    }
                ],
            },
        },
        "camera": {
            "positionRoomMm": camera_position_room_mm
            or {"x": 9200, "y": 4300, "z": 2650},
            "targetRoomMm": target_room_mm or {"x": 3000, "y": 2000, "z": 400},
            "fovDeg": 45,
            "aspect": 2.2797,
            "near": 0.01,
            "far": 500,
        },
        "totalObjectCount": len(objects),
        "visibleObjectCount": len(objects),
        "allObjects": objects,
        "visibleObjects": objects,
    }


class SnapshotPromptCompilerArchitectureTest(unittest.TestCase):
    def test_compile_snapshot_prompt_drops_minor_decor_when_disabled(self) -> None:
        result = snapshot_prompt_compiler.compile_snapshot_prompt(
            _snapshot(),
            config=snapshot_prompt_compiler.SnapshotPromptCompilerConfig(
                include_minor_decor=False
            ),
        )

        dropped_ids = {
            item["id"]
            for item in result["scene_ir"]["minor_objects_dropped"]  # type: ignore[index]
        }
        included_ids = {
            item["id"]
            for item in result["scene_ir"]["minor_objects_included"]  # type: ignore[index]
        }

        self.assertIn("vase_1", dropped_ids)
        self.assertIn("wall_art_1", dropped_ids)
        self.assertFalse(included_ids)
        self.assertIn(
            "Minor decor may be omitted if it does not change the composition",
            " ".join(result["layout_constraints"]),
        )

    def test_compile_snapshot_prompt_keeps_minor_decor_when_enabled(self) -> None:
        result = snapshot_prompt_compiler.compile_snapshot_prompt(
            _snapshot(),
            config=snapshot_prompt_compiler.SnapshotPromptCompilerConfig(
                include_minor_decor=True
            ),
        )

        included_ids = {
            item["id"]
            for item in result["scene_ir"]["minor_objects_included"]  # type: ignore[index]
        }

        self.assertIn("vase_1", included_ids)
        self.assertIn("wall_art_1", included_ids)
        self.assertNotIn(
            "Minor decor may be omitted if it does not change the composition",
            " ".join(result["layout_constraints"]),
        )

    def test_compile_snapshot_prompt_extracts_occlusion_and_top_down_shape(
        self,
    ) -> None:
        result = snapshot_prompt_compiler.compile_snapshot_prompt(
            _snapshot(
                polygon_mm=[
                    {"x": 0, "y": 0},
                    {"x": 6000, "y": 0},
                    {"x": 6000, "y": 2200},
                    {"x": 3600, "y": 2200},
                    {"x": 3600, "y": 4000},
                    {"x": 0, "y": 4000},
                ],
                camera_position_room_mm={"x": 3000, "y": 2000, "z": 9000},
                target_room_mm={"x": 3000, "y": 2000, "z": 0},
            ),
            config=snapshot_prompt_compiler.SnapshotPromptCompilerConfig(
                include_minor_decor=False
            ),
        )

        room_shell = result["scene_ir"]["room_shell"]  # type: ignore[index]
        occlusions = result["scene_ir"]["occlusion_relations"]  # type: ignore[index]

        self.assertEqual(room_shell["canonical_shape"], "L_shaped")
        self.assertEqual(room_shell["visible_shell_mode"], "canonical")
        self.assertTrue(occlusions)
        self.assertEqual(occlusions[0]["occluder_id"], "tv_console")
        self.assertEqual(occlusions[0]["occluded_id"], "sectional_sofa")
        self.assertIn("L-shaped", result["primary_prompt"])
        self.assertIn("occludes", result["primary_prompt"])

    def test_compile_snapshot_prompt_emits_detailed_layout_facts(self) -> None:
        detailed_snapshot = _snapshot(
            polygon_mm=[
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 3000},
                {"x": 3500, "y": 3000},
                {"x": 3500, "y": 4000},
                {"x": 0, "y": 4000},
            ],
            camera_position_room_mm={"x": 3000, "y": 2000, "z": 9000},
            target_room_mm={"x": 3000, "y": 2000, "z": 0},
            visible_objects=[
                _visible_object(
                    object_id="bookshelf_1",
                    object_type="bookshelf",
                    distance_to_camera_m=6.3,
                    screen_bbox_px={"minX": 90, "minY": 410, "maxX": 190, "maxY": 560},
                    color_hex="#46392B",
                    material="wood",
                    width_mm=620,
                    depth_mm=280,
                    height_mm=1800,
                    plan_position_mm={"x": 450, "y": 450, "z": 900},
                    plan_rotation_deg=270,
                ),
                _visible_object(
                    object_id="storage_cabinet_1",
                    object_type="storage_cabin",
                    distance_to_camera_m=5.9,
                    screen_bbox_px={"minX": 220, "minY": 430, "maxX": 340, "maxY": 560},
                    color_hex="#8A7866",
                    material="wood",
                    width_mm=720,
                    depth_mm=420,
                    height_mm=900,
                    plan_position_mm={"x": 980, "y": 430, "z": 450},
                    plan_rotation_deg=0,
                ),
                _visible_object(
                    object_id="console_table_1",
                    object_type="console_table",
                    distance_to_camera_m=5.5,
                    screen_bbox_px={"minX": 360, "minY": 430, "maxX": 520, "maxY": 560},
                    color_hex="#A18C73",
                    material="wood",
                    width_mm=1200,
                    depth_mm=420,
                    height_mm=780,
                    plan_position_mm={"x": 1800, "y": 450, "z": 390},
                    plan_rotation_deg=0,
                ),
                _visible_object(
                    object_id="side_table_1",
                    object_type="side_table",
                    distance_to_camera_m=5.3,
                    screen_bbox_px={"minX": 540, "minY": 440, "maxX": 610, "maxY": 540},
                    color_hex="#B7AA98",
                    material="wood",
                    width_mm=420,
                    depth_mm=420,
                    height_mm=540,
                    plan_position_mm={"x": 2550, "y": 520, "z": 270},
                    plan_rotation_deg=0,
                ),
                _visible_object(
                    object_id="floor_lamp_1",
                    object_type="floor_lamp",
                    distance_to_camera_m=5.2,
                    screen_bbox_px={"minX": 630, "minY": 420, "maxX": 680, "maxY": 560},
                    color_hex="#1D1A18",
                    material="metal",
                    width_mm=220,
                    depth_mm=220,
                    height_mm=1650,
                    plan_position_mm={"x": 3000, "y": 520, "z": 825},
                    plan_rotation_deg=0,
                ),
                _visible_object(
                    object_id="pet_bed_1",
                    object_type="pet_bed",
                    distance_to_camera_m=5.1,
                    screen_bbox_px={"minX": 690, "minY": 445, "maxX": 780, "maxY": 550},
                    color_hex="#DDD2C5",
                    material="fabric",
                    width_mm=620,
                    depth_mm=420,
                    height_mm=180,
                    plan_position_mm={"x": 3420, "y": 480, "z": 90},
                    plan_rotation_deg=0,
                ),
                _visible_object(
                    object_id="sofa_1",
                    object_type="sofa",
                    distance_to_camera_m=4.8,
                    screen_bbox_px={
                        "minX": 890,
                        "minY": 400,
                        "maxX": 1110,
                        "maxY": 560,
                    },
                    color_hex="#D8CCBC",
                    material="fabric",
                    width_mm=1900,
                    depth_mm=900,
                    height_mm=820,
                    plan_position_mm={"x": 5000, "y": 620, "z": 410},
                    plan_rotation_deg=0,
                ),
                _visible_object(
                    object_id="tv_console_1",
                    object_type="tv_console",
                    distance_to_camera_m=4.0,
                    screen_bbox_px={
                        "minX": 1060,
                        "minY": 260,
                        "maxX": 1180,
                        "maxY": 410,
                    },
                    color_hex="#343536",
                    material="wood",
                    width_mm=420,
                    depth_mm=1320,
                    height_mm=620,
                    plan_position_mm={"x": 5400, "y": 2200, "z": 310},
                    plan_rotation_deg=90,
                ),
                _visible_object(
                    object_id="media_shelf_1",
                    object_type="media_shelf",
                    distance_to_camera_m=4.1,
                    screen_bbox_px={
                        "minX": 910,
                        "minY": 295,
                        "maxX": 1010,
                        "maxY": 360,
                    },
                    color_hex="#C4C0BA",
                    material="wood",
                    width_mm=840,
                    depth_mm=260,
                    height_mm=520,
                    plan_position_mm={"x": 4480, "y": 2200, "z": 260},
                    plan_rotation_deg=90,
                ),
                _visible_object(
                    object_id="sectional_sofa_1",
                    object_type="sectional_sofa",
                    distance_to_camera_m=4.6,
                    screen_bbox_px={"minX": 505, "minY": 165, "maxX": 700, "maxY": 385},
                    color_hex="#D0C2B1",
                    material="fabric",
                    width_mm=2400,
                    depth_mm=820,
                    height_mm=860,
                    plan_position_mm={"x": 2900, "y": 2350, "z": 430},
                    plan_rotation_deg=270,
                ),
                _visible_object(
                    object_id="ottoman_1",
                    object_type="ottoman",
                    distance_to_camera_m=4.9,
                    screen_bbox_px={"minX": 410, "minY": 230, "maxX": 500, "maxY": 340},
                    color_hex="#D5C6B4",
                    material="fabric",
                    width_mm=640,
                    depth_mm=640,
                    height_mm=430,
                    plan_position_mm={"x": 2150, "y": 2350, "z": 215},
                    plan_rotation_deg=270,
                ),
                _visible_object(
                    object_id="armchair_1",
                    object_type="armchair",
                    distance_to_camera_m=5.0,
                    screen_bbox_px={"minX": 325, "minY": 165, "maxX": 450, "maxY": 335},
                    color_hex="#D4C8B9",
                    material="fabric",
                    width_mm=820,
                    depth_mm=760,
                    height_mm=930,
                    plan_position_mm={"x": 2150, "y": 3050, "z": 465},
                    plan_rotation_deg=270,
                ),
                _visible_object(
                    object_id="recliner_1",
                    object_type="recliner",
                    distance_to_camera_m=5.8,
                    screen_bbox_px={"minX": 130, "minY": 90, "maxX": 280, "maxY": 220},
                    color_hex="#CDC5BB",
                    material="fabric",
                    width_mm=1050,
                    depth_mm=860,
                    height_mm=980,
                    plan_position_mm={"x": 700, "y": 3550, "z": 490},
                    plan_rotation_deg=225,
                ),
            ],
        )

        result = snapshot_prompt_compiler.compile_snapshot_prompt(
            detailed_snapshot,
            config=snapshot_prompt_compiler.SnapshotPromptCompilerConfig(
                include_minor_decor=False
            ),
        )

        room_shell = result["scene_ir"]["room_shell"]  # type: ignore[index]
        strict_layout_facts = result["strict_layout_facts"]
        object_facts = result["scene_ir"]["object_facts"]  # type: ignore[index]
        spatial_relation_facts = result["scene_ir"]["spatial_relation_facts"]  # type: ignore[index]

        self.assertEqual(room_shell["canonical_shape"], "L_shaped")
        self.assertEqual(room_shell["missing_corner_compass"], "northeast")
        self.assertIn(
            "The room footprint is L-shaped with a missing northeast corner.",
            strict_layout_facts,
        )
        self.assertIn(
            "The window is on the south wall near the center.",
            strict_layout_facts,
        )
        self.assertTrue(
            any(
                "bookshelf is in the southwest corner, facing east toward the room center"
                in fact
                for fact in object_facts
            )
        )
        self.assertTrue(
            any(
                "TV console with integrated flat-screen TV is along the east wall, facing west toward the room center"
                in fact
                for fact in object_facts
            )
        )
        self.assertTrue(
            any(
                "positioned in front of the south-wall window" in fact
                for fact in object_facts
            )
        )
        self.assertIn(
            "The media shelf sits in front of the TV console with integrated flat-screen TV.",
            spatial_relation_facts,
        )
        self.assertIn(
            "The ottoman sits behind the sectional sofa.",
            spatial_relation_facts,
        )
        self.assertIn(
            "The armchair sits beside the ottoman.",
            spatial_relation_facts,
        )

    def test_compile_snapshot_prompt_from_path_reads_json_file(self) -> None:
        snapshot_path = None
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
            snapshot_path = Path(handle.name)
            json.dump(_snapshot(), handle)

        try:
            result = snapshot_prompt_compiler.compile_snapshot_prompt_from_path(
                snapshot_path
            )
        finally:
            if snapshot_path and snapshot_path.exists():
                snapshot_path.unlink()

        self.assertEqual(result["scene_ir"]["visible_object_count"], 6)

    def test_config_reads_minor_decor_env_flag(self) -> None:
        with mock.patch.dict(
            os.environ, {"TKNT_PROMPT_COMPILER_INCLUDE_MINOR_DECOR": "1"}
        ):
            config = snapshot_prompt_compiler.SnapshotPromptCompilerConfig.from_env()

        self.assertTrue(config.include_minor_decor)


if __name__ == "__main__":
    unittest.main()
