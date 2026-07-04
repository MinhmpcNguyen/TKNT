from __future__ import annotations

import base64
from io import BytesIO
import unittest

from PIL import Image

from pipeline.semantic_layout_locks import (
    semantic_layout_guide_for_snapshot_image,
    semantic_layout_lock_prompt_for_snapshot,
)


class SemanticLayoutLocksTest(unittest.TestCase):
    def test_prompt_includes_room_shell_and_opening_locks(self) -> None:
        prompt = semantic_layout_lock_prompt_for_snapshot(_snapshot_payload())

        self.assertIsNotNone(prompt)
        prompt_text = prompt or ""
        self.assertIn("Semantic layout locks", prompt_text)
        self.assertIn("Room-shell lock", prompt_text)
        self.assertIn("Opening lock", prompt_text)
        self.assertIn("Shadow-vs-geometry lock", prompt_text)

    def test_semantic_guide_matches_experiment_style_inputs(self) -> None:
        result = semantic_layout_guide_for_snapshot_image(
            snapshot_payload=_snapshot_payload(),
            source_image_data_url=_source_image_data_url(),
        )

        self.assertIsNotNone(result.prompt)
        self.assertTrue(result.image_data_url)
        self.assertEqual(result.metadata["guide_type"], "room_shell_semantic")
        self.assertEqual(result.metadata["opening_bbox_count"], 1)
        self.assertEqual(result.metadata["clean_object_count"], 1)
        self.assertGreater(result.metadata["semantic_surface_count"], 0)
        image = _decode_data_url_image(result.image_data_url or "")
        self.assertEqual(image.size, (300, 200))


def _snapshot_payload() -> dict[str, object]:
    return {
        "canvas": {"widthPx": 300, "heightPx": 200},
        "camera": {
            "aspect": 1.5,
            "far": 500,
            "fovDeg": 84,
            "near": 0.01,
            "positionM": {"x": 1.8345, "y": 2.62, "z": -1.8112},
            "targetM": {"x": -0.056, "y": 1.064, "z": -0.0391},
        },
        "room": {
            "boundsMm": {
                "maxX": 3950,
                "maxY": 4000,
                "minX": 0,
                "minY": 0,
            },
            "openings": {
                "doors": [
                    {
                        "screenBboxPx": {
                            "height": 50,
                            "maxX": 210,
                            "maxY": 110,
                            "minX": 175,
                            "minY": 60,
                            "width": 35,
                        }
                    }
                ],
                "windows": [],
            },
            "polygonMm": [
                {"x": 0, "y": 0},
                {"x": 3000, "y": 0},
                {"x": 3950, "y": 1000},
                {"x": 3950, "y": 3950},
                {"x": 0, "y": 4000},
            ],
        },
        "visibleObjects": [
            {
                "id": "bed",
                "type": "bed",
                "screenBboxPx": {
                    "height": 55,
                    "maxX": 230,
                    "maxY": 170,
                    "minX": 140,
                    "minY": 115,
                    "width": 90,
                },
            }
        ],
    }


def _source_image_data_url() -> str:
    image = Image.new("RGB", (300, 200), (230, 226, 216))
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", maxsplit=1)
    return Image.open(BytesIO(base64.b64decode(encoded))).copy()


if __name__ == "__main__":
    unittest.main()
