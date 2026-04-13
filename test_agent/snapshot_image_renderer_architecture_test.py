from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.pop("pipeline.snapshot_image_renderer", None)

module_path = REPO_ROOT / "pipeline" / "snapshot_image_renderer.py"
spec = importlib.util.spec_from_file_location(
    "pipeline.snapshot_image_renderer",
    module_path,
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load pipeline.snapshot_image_renderer for tests.")
snapshot_image_renderer = importlib.util.module_from_spec(spec)
sys.modules["pipeline.snapshot_image_renderer"] = snapshot_image_renderer
spec.loader.exec_module(snapshot_image_renderer)

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS8cAAAAASUVORK5CYII="
)


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
) -> dict[str, object]:
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
        "planPositionMm": {"x": 1000, "y": 1000, "z": height_mm / 2.0},
        "dimensionsMm": {
            "width": width_mm,
            "depth": depth_mm,
            "height": height_mm,
        },
        "planRotationDeg": 0,
        "sceneYawDeg": 0,
        "placeOn": None,
        "distanceToCameraM": distance_to_camera_m,
        "screenCenterPx": {
            "x": (screen_bbox_px["minX"] + screen_bbox_px["maxX"]) / 2.0,
            "y": (screen_bbox_px["minY"] + screen_bbox_px["maxY"]) / 2.0,
        },
        "screenBboxPx": {
            **screen_bbox_px,
            "width": screen_bbox_px["maxX"] - screen_bbox_px["minX"],
            "height": screen_bbox_px["maxY"] - screen_bbox_px["minY"],
        },
        "visibleSampleCount": 4,
        "visibleSampleFraction": 0.444,
    }


def _snapshot() -> dict[str, object]:
    objects = [
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
    ]
    return {
        "capturedAt": "2026-04-10T03:00:00.000Z",
        "variantId": "variant_1",
        "canvas": {"widthPx": 1345, "heightPx": 590},
        "room": {
            "boundsMm": {"minX": 0, "minY": 0, "maxX": 6000, "maxY": 4000},
            "polygonMm": [
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
            "openings": {"doors": [], "windows": []},
        },
        "camera": {
            "positionRoomMm": {"x": 9200, "y": 4300, "z": 2650},
            "targetRoomMm": {"x": 3000, "y": 2000, "z": 400},
            "fovDeg": 45,
            "aspect": 2.2,
            "near": 0.01,
            "far": 500,
        },
        "totalObjectCount": len(objects),
        "visibleObjectCount": len(objects),
        "allObjects": objects,
        "visibleObjects": objects,
    }


def _tiny_png_data_url() -> str:
    return f"data:image/png;base64,{_TINY_PNG_BASE64}"


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeHttpClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        self.calls: list[dict[str, object]] = []

    def __enter__(self) -> _FakeHttpClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb
        return None

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> _FakeHttpResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "images": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _tiny_png_data_url()},
                                }
                            ]
                        }
                    }
                ]
            }
        )


class SnapshotImageRendererArchitectureTest(unittest.TestCase):
    def test_render_snapshot_image_posts_openrouter_image_request(self) -> None:
        fake_http_client = _FakeHttpClient()
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                layout_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="google/gemini-2.5-flash-image",
                    image_size="1K",
                    max_tokens=128,
                    openrouter_base_url="https://openrouter.ai/api/v1",
                    openrouter_api_key="secret",
                    openrouter_site_url="https://example.com",
                    openrouter_site_name="TKNT",
                    include_layout_2d_reference=True,
                ),
            )

        self.assertTrue(
            result["image"]["data_url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(
            result["models"]["image_model_name"], "google/gemini-2.5-flash-image"
        )
        self.assertEqual(
            result["request"]["user_prompt"],
            "Use warm walnut materials and soft evening lighting.",
        )
        self.assertEqual(result["request"]["aspect_ratio"], "21:9")
        self.assertTrue(result["request"]["layout_reference_enabled"])
        self.assertTrue(result["request"]["layout_reference_used"])

        request_payload = fake_http_client.calls[0]["json"]
        self.assertEqual(
            request_payload["model"], "google/gemini-2.5-flash-image"
        )
        self.assertEqual(request_payload["max_tokens"], 128)
        self.assertEqual(request_payload["modalities"], ["image"])
        self.assertEqual(
            request_payload["image_config"]["aspect_ratio"],
            "21:9",
        )
        self.assertEqual(
            request_payload["messages"][0]["content"],
            snapshot_image_renderer._GEMINI_SYSTEM_PROMPT,
        )
        user_content = request_payload["messages"][1]["content"]
        self.assertEqual(
            user_content[0]["text"],
            "Use warm walnut materials and soft evening lighting.",
        )
        self.assertEqual(user_content[1]["image_url"]["url"], _tiny_png_data_url())
        self.assertEqual(user_content[2]["image_url"]["url"], _tiny_png_data_url())

    def test_render_config_reads_openrouter_env_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "secret",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_SITE_URL": "https://example.com",
                "OPENROUTER_SITE_NAME": "TKNT",
                "TKNT_SNAPSHOT_RENDER_IMAGE_MODEL": "google/gemini-2.5-flash-image",
                "TKNT_SNAPSHOT_RENDER_IMAGE_SIZE": "2K",
                "TKNT_SNAPSHOT_RENDER_MAX_TOKENS": "192",
                "TKNT_SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE": "true",
            },
        ):
            config = snapshot_image_renderer.SnapshotImageRenderConfig.from_env()

        self.assertEqual(config.image_model_name, "google/gemini-2.5-flash-image")
        self.assertEqual(config.image_size, "2K")
        self.assertEqual(config.max_tokens, 192)
        self.assertEqual(config.openrouter_api_key, "secret")
        self.assertEqual(config.openrouter_site_url, "https://example.com")
        self.assertEqual(config.openrouter_site_name, "TKNT")
        self.assertTrue(config.include_layout_2d_reference)


if __name__ == "__main__":
    unittest.main()
