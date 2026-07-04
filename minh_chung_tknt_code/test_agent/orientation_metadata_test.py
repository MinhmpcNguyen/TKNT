from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout.orientation_metadata import (  # noqa: E402
    enrich_normalized_layout_orientations,
    enrich_styled_output_orientations,
)


class OrientationMetadataTest(unittest.TestCase):
    def test_enriches_styled_output_from_absolute_layout(self) -> None:
        enriched = enrich_styled_output_orientations(
            styled_payload={
                "objects": [
                    {
                        "instance_id": "bed",
                        "bbox": {
                            "min_x": 0,
                            "min_y": 0,
                            "max_x": 1200,
                            "max_y": 2000,
                        },
                    }
                ]
            },
            absolute_layout_payload={
                "objects": [
                    {
                        "object_id": "bed",
                        "rotation_ccw": 270,
                        "front_world": {"dx": -1, "dy": 0},
                        "front_side_world": "left",
                    }
                ]
            },
        )

        bed = enriched["objects"][0]
        self.assertEqual(bed["rotation_ccw"], 270)
        self.assertEqual(bed["front_world"], {"dx": -1, "dy": 0})
        self.assertEqual(bed["front_side_world"], "left")

    def test_enriches_normalized_object_and_raw_payload(self) -> None:
        enriched = enrich_normalized_layout_orientations(
            normalized_payload={
                "objects": [
                    {
                        "id": "wardrobe",
                        "rotation_degrees": 0.0,
                        "raw": {"instance_id": "wardrobe"},
                    }
                ]
            },
            absolute_layout_payload={
                "objects": [
                    {
                        "object_id": "wardrobe",
                        "rotation_ccw": 90,
                        "front_world": {"dx": 1, "dy": 0},
                        "front_side_world": "right",
                        "axis_world": {"dx": 0, "dy": 1},
                    }
                ]
            },
        )

        wardrobe = enriched["objects"][0]
        self.assertEqual(wardrobe["rotation_degrees"], 90.0)
        self.assertEqual(wardrobe["front_world"], {"dx": 1, "dy": 0})
        self.assertEqual(wardrobe["front_side_world"], "right")
        self.assertEqual(wardrobe["raw"]["front_world"], {"dx": 1, "dy": 0})


if __name__ == "__main__":
    unittest.main()
