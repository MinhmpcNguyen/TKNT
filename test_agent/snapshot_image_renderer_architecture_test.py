from __future__ import annotations

import base64
import importlib.util
import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import httpx
from PIL import Image


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

_TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS8cAAAAASUVORK5CYII="


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


def _snapshot_with_empty_visible_objects() -> dict[str, object]:
    payload = _snapshot()
    raw_objects = payload.get("allObjects")
    objects = list(raw_objects) if isinstance(raw_objects, list) else []
    objects.append(
        _visible_object(
            object_id="media_shelf",
            object_type="media_shelf",
            distance_to_camera_m=4.6,
            screen_bbox_px={"minX": 700, "minY": 260, "maxX": 820, "maxY": 520},
            color_hex="#8F7358",
            material="wood",
            width_mm=870,
            depth_mm=300,
            height_mm=1360,
        )
    )
    return {
        **payload,
        "totalObjectCount": len(objects),
        "visibleObjectCount": 0,
        "allObjects": objects,
        "visibleObjects": [],
    }


def _tiny_png_data_url() -> str:
    return f"data:image/png;base64,{_TINY_PNG_BASE64}"


def _image_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(
        "utf-8"
    )


def _inline_image_size(part: dict[str, object]) -> tuple[int, int]:
    inline_data = part.get("inline_data")
    if not isinstance(inline_data, dict):
        raise AssertionError("Expected inline_data payload.")
    image_base64 = inline_data.get("data")
    if not isinstance(image_base64, str):
        raise AssertionError("Expected inline image data.")
    image_bytes = base64.b64decode(image_base64, validate=True)
    with Image.open(BytesIO(image_bytes)) as image:
        return image.size


def _gemini_image_payload() -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": _TINY_PNG_BASE64,
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ]
    }


def _openai_image_payload() -> dict[str, object]:
    return {
        "created": 1,
        "data": [
            {
                "b64_json": _TINY_PNG_BASE64,
            }
        ],
    }


def _gemini_no_image_payload(
    *,
    finish_reason: str,
    finish_message: str,
) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {},
                "finishReason": finish_reason,
                "finishMessage": finish_message,
                "index": 0,
            }
        ]
    }


class _FakeHttpResponse:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com/v1beta")
            response = httpx.Response(
                self.status_code,
                request=request,
                text=self.text,
            )
            raise httpx.HTTPStatusError(
                f"Client error '{self.status_code}' for url '{request.url}'",
                request=request,
                response=response,
            )
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeHttpClient:
    def __init__(
        self,
        responses: list[dict[str, object] | _FakeHttpResponse],
    ) -> None:
        self._responses = list(responses)
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
        json: dict[str, object] | None = None,
        data: dict[str, object] | None = None,
        files: list[object] | None = None,
        headers: dict[str, str],
    ) -> _FakeHttpResponse:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "data": data,
                "files": files,
                "headers": headers,
            }
        )
        if not self._responses:
            raise AssertionError("No fake Gemini response configured.")
        response = self._responses.pop(0)
        if isinstance(response, _FakeHttpResponse):
            return response
        return _FakeHttpResponse(response)


class SnapshotImageRendererArchitectureTest(unittest.TestCase):
    def test_openai_generate_sends_geometry_references_in_prompt_matched_order(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with (
            mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx,
            mock.patch.object(
                snapshot_image_renderer,
                "_validate_generated_geometry_against_reference",
                return_value={"status": "passed", "reason": "test_pass"},
            ),
        ):
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                preset_selection={
                    "style": "modern",
                    "lighting": "daylight_1",
                    "scenery": "none",
                },
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        self.assertEqual(
            result["image"]["request_strategy"],
            "openai_edit_geometry_locked",
        )
        self.assertEqual(result["image"]["retry_count"], 0)
        self.assertEqual(result["metadata"]["validation"]["status"], "passed")
        self.assertEqual(
            result["request"]["geometry_references_used"],
            {
                "depth_map": True,
                "edge_map": True,
                "segmentation_map": True,
            },
        )
        self.assertEqual(
            result["request"]["layout_lock_mode"],
            "strict_generate_layout_lock",
        )
        self.assertGreaterEqual(len(result["request"]["locked_items"]), 2)

        request_call = fake_http_client.calls[0]
        self.assertEqual(request_call["url"], "https://example.com/v1/images/edits")
        files = request_call["files"]
        self.assertIsInstance(files, list)
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(
            image_file_names,
            [
                "source.png",
                "depth_map.png",
                "edge_map.png",
                "segmentation_map.png",
                "edit_guide.png",
            ],
        )
        prompt = request_call["data"]["prompt"]
        self.assertIn("Image 1 is the clean 3D source render", prompt)
        self.assertIn("Image 2 is a deterministic depth map", prompt)
        self.assertIn("Image 3 is a deterministic edge map", prompt)
        self.assertIn("Image 4 is a deterministic segmentation map", prompt)
        self.assertIn("Image 5 is a same-camera labeled structural guide", prompt)
        self.assertIn("Locked visible object and opening map", prompt)
        self.assertIn("object tv_console", prompt)
        self.assertNotIn("Minor decor may be omitted", prompt)
        self.assertIn(
            "Apply these selected presets as finish realism, material texture, "
            "lighting/illumination, and exterior scenery visible through existing "
            "openings only.",
            prompt,
        )

    def test_openai_generate_fallback_locks_all_objects_when_visible_list_is_empty(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with (
            mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx,
            mock.patch.object(
                snapshot_image_renderer,
                "_validate_generated_geometry_against_reference",
                return_value={"status": "passed", "reason": "test_pass"},
            ),
        ):
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot_with_empty_visible_objects(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        prompt = fake_http_client.calls[0]["data"]["prompt"]
        locked_item_ids = {item.get("id") for item in result["request"]["locked_items"]}
        self.assertNotIn("There are 0 visible objects in frame", prompt)
        self.assertIn("object tv_console", prompt)
        self.assertIn("object media_shelf", prompt)
        self.assertIn("tv_console", locked_item_ids)
        self.assertIn("media_shelf", locked_item_ids)
        self.assertEqual(
            result["request"]["visibility_diagnostics"]["status"],
            "degraded",
        )
        self.assertIn(
            "visible_objects_empty_with_all_objects_present",
            result["request"]["visibility_diagnostics"]["reasons"],
        )

    def test_openai_generate_sends_object_geometry_lock_references(self) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with (
            mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx,
            mock.patch.object(
                snapshot_image_renderer,
                "_validate_generated_geometry_against_reference",
                return_value={"status": "passed", "reason": "test_pass"},
            ),
        ):
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                locked_geometry_references=[
                    snapshot_image_renderer.SnapshotLockedGeometryReference(
                        object_id="tv_console",
                        object_name="TV console",
                        image_data_url=_tiny_png_data_url(),
                        source="clean_3d_object_crop",
                        screen_bbox_px={
                            "minX": 500,
                            "minY": 300,
                            "maxX": 650,
                            "maxY": 430,
                            "width": 150,
                            "height": 130,
                        },
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        files = fake_http_client.calls[0]["files"]
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(
            image_file_names,
            [
                "source.png",
                "depth_map.png",
                "edge_map.png",
                "segmentation_map.png",
                "edit_guide.png",
                "geometry_lock_1.png",
            ],
        )
        prompt = fake_http_client.calls[0]["data"]["prompt"]
        self.assertIn("Object geometry lock references:", prompt)
        self.assertIn("Geometry lock reference 1: object id tv_console", prompt)
        self.assertEqual(
            result["request"]["locked_geometry_references"][0]["object_id"],
            "tv_console",
        )

    def test_openai_generate_sends_object_geometry_atlas_for_arbitrary_objects(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with (
            mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx,
            mock.patch.object(
                snapshot_image_renderer,
                "_validate_generated_geometry_against_reference",
                return_value={"status": "passed", "reason": "test_pass"},
            ),
        ):
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                locked_geometry_atlases=[
                    snapshot_image_renderer.SnapshotLockedGeometryAtlas(
                        atlas_index=1,
                        image_data_url=_tiny_png_data_url(),
                        source="clean_3d_visible_object_atlas",
                        objects=[
                            snapshot_image_renderer.SnapshotLockedGeometryAtlasObject(
                                object_id="sectional_sofa",
                                object_name="sectional sofa",
                                atlas_index=1,
                                cell_index=1,
                                cell_label="G1",
                                screen_bbox_px={
                                    "minX": 470,
                                    "minY": 245,
                                    "maxX": 790,
                                    "maxY": 450,
                                    "width": 320,
                                    "height": 205,
                                },
                                cell_bbox_norm={
                                    "x": 0.02,
                                    "y": 0.08,
                                    "width": 0.2,
                                    "height": 0.18,
                                },
                            ),
                            snapshot_image_renderer.SnapshotLockedGeometryAtlasObject(
                                object_id="tv_console",
                                object_name="TV console",
                                atlas_index=1,
                                cell_index=2,
                                cell_label="G2",
                                screen_bbox_px={
                                    "minX": 500,
                                    "minY": 300,
                                    "maxX": 650,
                                    "maxY": 430,
                                    "width": 150,
                                    "height": 130,
                                },
                                cell_bbox_norm={
                                    "x": 0.27,
                                    "y": 0.08,
                                    "width": 0.18,
                                    "height": 0.16,
                                },
                            ),
                        ],
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        files = fake_http_client.calls[0]["files"]
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(
            image_file_names,
            [
                "source.png",
                "depth_map.png",
                "edge_map.png",
                "segmentation_map.png",
                "edit_guide.png",
                "geometry_atlas_1.png",
            ],
        )
        prompt = fake_http_client.calls[0]["data"]["prompt"]
        self.assertIn("Object geometry lock references:", prompt)
        self.assertIn("Geometry atlas 1:", prompt)
        self.assertIn("G1: object id sectional_sofa", prompt)
        self.assertIn("G2: object id tv_console", prompt)
        self.assertEqual(result["request"]["locked_geometry_atlas_count"], 1)
        self.assertEqual(result["request"]["locked_geometry_object_count"], 2)
        self.assertEqual(
            result["request"]["geometry_lock_policy"],
            "all_visible_objects_atlas",
        )

    def test_frontend_geometry_atlas_builder_has_no_semantic_whitelist(self) -> None:
        source = (
            REPO_ROOT / "frontend" / "src" / "components" / "ThreeDPreviewPanel.tsx"
        ).read_text()
        self.assertNotIn("GEOMETRY_LOCK_OBJECT_TYPE_HINTS", source)
        self.assertNotIn("shouldLockObjectGeometry", source)
        self.assertIn("buildLockedGeometryAtlases", source)

    def test_openai_geometry_validation_failure_retries_once(self) -> None:
        fake_http_client = _FakeHttpClient(
            [_openai_image_payload(), _openai_image_payload()]
        )
        with (
            mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx,
            mock.patch.object(
                snapshot_image_renderer,
                "_validate_generated_geometry_against_reference",
                side_effect=[
                    {
                        "status": "failed",
                        "reason": "edge_alignment_score_below_threshold",
                    },
                    {
                        "status": "passed",
                        "reason": "geometry_reference_alignment_acceptable",
                    },
                ],
            ),
        ):
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        self.assertEqual(len(fake_http_client.calls), 2)
        self.assertEqual(
            result["image"]["request_strategy"],
            "openai_edit_geometry_retry",
        )
        self.assertEqual(result["image"]["retry_count"], 1)
        self.assertEqual(result["metadata"]["validation"]["status"], "passed")
        retry_prompt = fake_http_client.calls[1]["data"]["prompt"]
        self.assertIn("Geometry retry instruction:", retry_prompt)
        self.assertIn("edge_alignment_score_below_threshold", retry_prompt)

    def test_locked_item_shape_validation_fails_for_missing_tv_console_edges(
        self,
    ) -> None:
        reference = Image.new("RGB", (96, 64), "white")
        reference_pixels = reference.load()
        for x in range(10, 86):
            reference_pixels[x, 48] = (0, 0, 0)
        for y in range(18, 49):
            reference_pixels[10, y] = (0, 0, 0)
            reference_pixels[52, y] = (0, 0, 0)
            reference_pixels[86, y] = (0, 0, 0)
        generated = Image.new("RGB", (96, 64), "black")
        validation = snapshot_image_renderer._validate_locked_item_shape(
            generated_image=generated,
            locked_items=[
                {
                    "id": "tv_console",
                    "type": "tv_console",
                    "screen_bbox_px": {
                        "minX": 0,
                        "minY": 0,
                        "maxX": 96,
                        "maxY": 64,
                        "width": 96,
                        "height": 64,
                    },
                    "canvas_width_px": 96,
                    "canvas_height_px": 64,
                }
            ],
            locked_geometry_references=[
                snapshot_image_renderer.SnapshotLockedGeometryReference(
                    object_id="tv_console",
                    object_name="TV console",
                    image_data_url=_image_data_url(reference),
                    screen_bbox_px={
                        "minX": 0,
                        "minY": 0,
                        "maxX": 96,
                        "maxY": 64,
                        "width": 96,
                        "height": 64,
                    },
                )
            ],
            locked_geometry_atlases=[],
        )

        self.assertEqual(validation["status"], "failed")
        self.assertEqual(validation["failed_items"][0]["id"], "tv_console")

    def test_openai_generate_without_geometry_maps_keeps_legacy_strategy(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        self.assertEqual(result["image"]["request_strategy"], "openai_edit")
        self.assertEqual(result["image"]["retry_count"], 0)
        self.assertEqual(
            result["request"]["geometry_references_used"],
            {
                "depth_map": False,
                "edge_map": False,
                "segmentation_map": False,
            },
        )
        files = fake_http_client.calls[0]["files"]
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(image_file_names, ["source.png", "edit_guide.png"])

    def test_build_gemini_generation_config_uses_image_response_for_gemini_3_pro_preview(
        self,
    ) -> None:
        config = snapshot_image_renderer._build_gemini_generation_config(
            model_name="gemini-3-pro-image-preview",
            image_size="2K",
            max_output_tokens=192,
            aspect_ratio="21:9",
        )

        self.assertEqual(
            config,
            {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": "21:9",
                    "imageSize": "2K",
                },
            },
        )

    def test_render_snapshot_image_uses_configured_request_shape_for_gemini_31_preview(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                layout_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                    include_layout_2d_reference=True,
                ),
            )

        self.assertTrue(
            result["image"]["data_url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(
            result["image"]["request_strategy"],
            "configured",
        )
        self.assertEqual(
            result["models"]["image_model_name"],
            "gemini-3.1-flash-image-preview",
        )
        self.assertEqual(result["request"]["aspect_ratio"], "21:9")
        self.assertTrue(result["request"]["layout_reference_enabled"])
        self.assertTrue(result["request"]["layout_reference_used"])

        request_payload = fake_http_client.calls[0]["json"]
        self.assertEqual(
            request_payload["generationConfig"],
            {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": "21:9",
                    "imageSize": "1K",
                },
            },
        )
        self.assertNotIn("systemInstruction", request_payload)
        user_parts = request_payload["contents"][0]["parts"]
        text_part = user_parts[0]["text"]
        self.assertIn(snapshot_image_renderer._GENERATE_SYSTEM_PROMPT, text_part)
        self.assertIn(
            "Use warm walnut materials and soft evening lighting.",
            text_part,
        )
        self.assertIn("Layout lock instructions:", text_part)
        self.assertIn(
            "Material, lighting, and scenery goal, secondary to the exact input view and locked palette:",
            text_part,
        )
        self.assertNotIn("Target visible object screen map", text_part)
        self.assertNotIn("screen bbox", text_part)
        self.assertEqual(_inline_image_size(user_parts[1]), (1584, 672))
        self.assertEqual(_inline_image_size(user_parts[2]), (1584, 672))

    def test_annotated_reference_replaces_top_down_layout_reference_for_generate(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                layout_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                    include_layout_2d_reference=True,
                ),
            )

        self.assertTrue(result["request"]["annotated_reference_used"])
        self.assertFalse(result["request"]["layout_reference_used"])
        request_payload = fake_http_client.calls[0]["json"]
        user_parts = request_payload["contents"][0]["parts"]
        text_part = user_parts[0]["text"]
        self.assertEqual(len(user_parts), 3)
        self.assertIn(
            "Image 2 is a same-camera labeled structural guide",
            text_part,
        )
        self.assertIn("Do not render the guide lines", text_part)

    def test_generate_can_use_full_scene_reference_with_target_layout_source(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                scene_reference_image_data_url=_tiny_png_data_url(),
                scene_reference_mode="target_layout_with_scene_reference",
                annotated_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Keep the loaded render identity from A1.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(
            result["request"]["scene_reference_mode"],
            "target_layout_with_scene_reference",
        )
        self.assertTrue(result["request"]["scene_reference_used"])
        self.assertFalse(result["request"]["reference_only_camera_transfer_used"])
        request_payload = fake_http_client.calls[0]["json"]
        user_parts = request_payload["contents"][0]["parts"]
        text_part = user_parts[0]["text"]
        self.assertEqual(len(user_parts), 4)
        self.assertIn(
            "Image 2 is a previously rendered same-room reference",
            text_part,
        )
        self.assertIn(
            "Image 3 is a same-camera labeled structural guide",
            text_part,
        )

    def test_generate_can_use_reference_only_camera_transfer_mode(self) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                scene_reference_image_data_url=_tiny_png_data_url(),
                scene_reference_mode="scene_reference_camera_transfer",
                annotated_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Move A1 to the newly selected camera.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(
            result["request"]["scene_reference_mode"],
            "scene_reference_camera_transfer",
        )
        self.assertFalse(result["request"]["scene_reference_used"])
        self.assertTrue(result["request"]["reference_only_camera_transfer_used"])
        self.assertTrue(result["request"]["annotated_reference_used"])
        request_payload = fake_http_client.calls[0]["json"]
        user_parts = request_payload["contents"][0]["parts"]
        text_part = user_parts[0]["text"]
        self.assertEqual(len(user_parts), 3)
        self.assertIn(
            "Image 1 is a previously rendered source reference",
            text_part,
        )
        self.assertIn("Image 2 is a same-camera labeled structural guide", text_part)
        self.assertIn("Target camera metadata:", text_part)
        self.assertNotIn("Target visible object screen map", text_part)

    def test_generate_can_pair_object_appearance_memory_references(self) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                edit_operations=[
                    snapshot_image_renderer.SnapshotEditOperation(
                        object_id="tv_console",
                        object_name="TV console",
                        replacement_image_data_url=_tiny_png_data_url(),
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(result["request"]["reference_image_count"], 1)
        request_payload = fake_http_client.calls[0]["json"]
        user_parts = request_payload["contents"][0]["parts"]
        text_part = user_parts[0]["text"]
        self.assertEqual(len(user_parts), 3)
        self.assertIn("Object appearance memory:", text_part)
        self.assertIn(
            "Reference image 1: object id tv_console",
            text_part,
        )
        self.assertEqual(_inline_image_size(user_parts[2]), (1584, 672))

    def test_edit_replacement_locks_selected_geometry_over_reference_scale(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                edit_source_image_data_url=_tiny_png_data_url(),
                render_mode="edit",
                edit_operations=[
                    snapshot_image_renderer.SnapshotEditOperation(
                        object_id="sectional_sofa",
                        object_name="sectional sofa",
                        replacement_image_data_url=_tiny_png_data_url(),
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(result["request"]["render_mode"], "edit")
        self.assertEqual(result["request"]["reference_image_count"], 1)
        request_payload = fake_http_client.calls[0]["json"]
        user_parts = request_payload["contents"][0]["parts"]
        text_part = user_parts[0]["text"]
        self.assertEqual(len(user_parts), 3)
        self.assertIn(
            "Selected object geometry locks from the source image",
            text_part,
        )
        self.assertIn(
            "override reference-image scale, crop, camera, and composition",
            text_part,
        )
        self.assertIn('"screenBboxPx"', text_part)
        self.assertIn(
            "appearance-only replacement reference, not a scale, camera, crop, layout, or composition reference",
            text_part,
        )
        self.assertIn(
            "Do not cover, remove, redraw, or relocate any unselected object.",
            text_part,
        )
        self.assertEqual(_inline_image_size(user_parts[1]), (1584, 672))
        self.assertEqual(_inline_image_size(user_parts[2]), (1584, 672))

    def test_edit_rejects_layout_transfer_without_an_actual_edit(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Provide at least one edit operation, preset change, or prompt change.",
        ):
            snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                layout_reference_image_data_url=_tiny_png_data_url(),
                edit_source_image_data_url=_tiny_png_data_url(),
                render_mode="edit",
                user_prompt="",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

    def test_camera_rerender_uses_new_3d_geometry_and_previous_render_as_memory(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with (
            mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx,
            mock.patch.object(
                snapshot_image_renderer,
                "_validate_generated_geometry_against_reference",
                return_value={"status": "passed", "reason": "test_pass"},
            ),
        ):
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                scene_reference_image_data_url=_tiny_png_data_url(),
                scene_reference_mode="target_layout_with_scene_reference",
                render_mode="camera_rerender",
                edit_operations=[
                    snapshot_image_renderer.SnapshotEditOperation(
                        object_id="sectional_sofa",
                        object_name="sectional sofa",
                        replacement_image_data_url=_tiny_png_data_url(),
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_provider="openai",
                    openai_api_key="secret",
                    openai_image_model="gpt-image-1.5",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                    openai_base_url="https://example.com/v1",
                ),
            )

        self.assertEqual(result["request"]["render_mode"], "camera_rerender")
        self.assertEqual(
            result["request"]["layout_lock_mode"],
            "camera_rerender_layout_lock",
        )
        self.assertEqual(result["request"]["reference_image_count"], 1)
        self.assertEqual(len(result["request"]["object_memory_used"]), 1)
        request_call = fake_http_client.calls[0]
        files = request_call["files"]
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(
            image_file_names,
            [
                "source.png",
                "depth_map.png",
                "edge_map.png",
                "segmentation_map.png",
                "scene_reference.png",
                "edit_guide.png",
                "reference_1.png",
            ],
        )
        prompt = request_call["data"]["prompt"]
        self.assertIn(
            "Image 1 is the clean 3D target render from the newly selected camera",
            prompt,
        )
        self.assertIn(
            "Use the previous photorealistic render and object reference images only as appearance memory",
            prompt,
        )
        self.assertIn("Object appearance memory:", prompt)

    def test_render_snapshot_image_applies_generation_config_for_gemini_31_preview(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_gemini_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(len(fake_http_client.calls), 1)
        self.assertIn("generationConfig", fake_http_client.calls[0]["json"])
        self.assertEqual(
            result["image"]["request_strategy"],
            "configured",
        )

    def test_render_snapshot_image_retries_without_generation_config_for_gemini_31_http_400(
        self,
    ) -> None:
        error_payload = {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        }
        fake_http_client = _FakeHttpClient(
            [
                _FakeHttpResponse(
                    error_payload,
                    status_code=400,
                    text='{"error":{"code":400,"message":"Request contains an invalid argument.","status":"INVALID_ARGUMENT"}}',
                ),
                _gemini_image_payload(),
            ]
        )
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            mocked_httpx.HTTPStatusError = httpx.HTTPStatusError
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(len(fake_http_client.calls), 2)
        self.assertIn("generationConfig", fake_http_client.calls[0]["json"])
        self.assertNotIn("generationConfig", fake_http_client.calls[1]["json"])
        self.assertEqual(
            result["image"]["request_strategy"],
            "compatibility_no_generation_config",
        )

    def test_render_snapshot_image_retries_source_only_when_supplemental_images_fail(
        self,
    ) -> None:
        error_payload = {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        }
        error_text = (
            '{"error":{"code":400,"message":"Request contains an invalid argument.",'
            '"status":"INVALID_ARGUMENT"}}'
        )
        fake_http_client = _FakeHttpClient(
            [
                _FakeHttpResponse(error_payload, status_code=400, text=error_text),
                _FakeHttpResponse(error_payload, status_code=400, text=error_text),
                _gemini_image_payload(),
            ]
        )
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            mocked_httpx.HTTPStatusError = httpx.HTTPStatusError
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3.1-flash-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(len(fake_http_client.calls), 3)
        self.assertEqual(
            result["image"]["request_strategy"],
            "compatibility_source_only_no_generation_config",
        )
        source_only_payload = fake_http_client.calls[2]["json"]
        self.assertNotIn("generationConfig", source_only_payload)
        source_only_parts = source_only_payload["contents"][0]["parts"]
        self.assertEqual(len(source_only_parts), 2)
        self.assertNotIn(
            "Image 2 is a same-camera labeled structural guide",
            source_only_parts[0]["text"],
        )

    def test_render_snapshot_image_retries_without_generation_config_for_gemini_3_pro(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient(
            [
                _gemini_no_image_payload(
                    finish_reason="MALFORMED_FUNCTION_CALL",
                    finish_message="Malformed function call: call:google:image_gen{...}",
                ),
                _gemini_image_payload(),
            ]
        )
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                user_prompt="Use warm walnut materials and soft evening lighting.",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    image_model_name="gemini-3-pro-image-preview",
                    image_size="1K",
                    max_output_tokens=128,
                    gemini_base_url="https://example.com/v1beta",
                    gemini_api_key="secret",
                    image_provider="gemini",
                ),
            )

        self.assertEqual(len(fake_http_client.calls), 2)
        self.assertIn("generationConfig", fake_http_client.calls[0]["json"])
        self.assertNotIn("generationConfig", fake_http_client.calls[1]["json"])
        self.assertEqual(
            result["image"]["request_strategy"],
            "compatibility_no_generation_config",
        )

    def test_render_snapshot_image_surfaces_finish_reason_when_no_image_is_returned(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient(
            [
                _gemini_no_image_payload(
                    finish_reason="SAFETY",
                    finish_message="Blocked by safety filters.",
                )
            ]
        )
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            with self.assertRaisesRegex(
                ValueError,
                "finishReason=SAFETY",
            ) as exc_info:
                snapshot_image_renderer.render_snapshot_image(
                    _snapshot(),
                    snapshot_image_data_url=_tiny_png_data_url(),
                    user_prompt="Use warm walnut materials and soft evening lighting.",
                    config=snapshot_image_renderer.SnapshotImageRenderConfig(
                        image_model_name="gemini-2.5-flash-image",
                        image_size="1K",
                        max_output_tokens=128,
                        gemini_base_url="https://example.com/v1beta",
                        gemini_api_key="secret",
                        image_provider="gemini",
                    ),
                )

        self.assertIn(
            "finishMessage=Blocked by safety filters.", str(exc_info.exception)
        )

    def test_render_config_reads_gemini_env_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GEMINI_IMAGE_API_KEY": "secret",
                "GEMINI_IMAGE_BASE_URL": "https://example.com/v1beta",
                "TKNT_SNAPSHOT_RENDER_IMAGE_MODEL": "gemini-2.5-flash-image",
                "TKNT_SNAPSHOT_RENDER_IMAGE_SIZE": "2K",
                "TKNT_SNAPSHOT_RENDER_MAX_TOKENS": "192",
                "TKNT_SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE": "true",
            },
        ):
            config = snapshot_image_renderer.SnapshotImageRenderConfig.from_env()

        self.assertEqual(config.image_model_name, "gemini-2.5-flash-image")
        self.assertEqual(config.image_size, "2K")
        self.assertEqual(config.max_output_tokens, 192)
        self.assertEqual(config.gemini_base_url, "https://example.com/v1beta")
        self.assertEqual(config.gemini_api_key, "secret")
        self.assertTrue(config.include_layout_2d_reference)


if __name__ == "__main__":
    unittest.main()
