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


def _openai_image_payload() -> dict[str, object]:
    return {
        "created": 1,
        "data": [
            {
                "b64_json": _TINY_PNG_BASE64,
            }
        ],
    }


def _openai_stream_chunks() -> list[str]:
    partial_payload = {
        "type": "image_edit.partial_image",
        "partial_image_index": 1,
        "b64_json": _TINY_PNG_BASE64,
    }
    completed_payload = {
        "type": "image_edit.completed",
        "b64_json": _TINY_PNG_BASE64,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }
    return [
        f"data: {json_dump(partial_payload)}\n",
        "\n",
        f"data: {json_dump(completed_payload)}\n\n",
    ]


def json_dump(payload: dict[str, object]) -> str:
    return snapshot_image_renderer.json.dumps(payload)


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
            request = httpx.Request("POST", "https://api.openai.com/v1/images/edits")
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


class _FakeStreamResponse:
    def __init__(
        self,
        chunks: list[str],
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self.text = text

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.openai.com/v1/images/edits")
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

    def iter_text(self):
        yield from self._chunks


class _FakeHttpClient:
    def __init__(
        self,
        responses: list[dict[str, object] | _FakeHttpResponse | _FakeStreamResponse],
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
            raise AssertionError("No fake OpenAI image response configured.")
        response = self._responses.pop(0)
        if isinstance(response, _FakeHttpResponse):
            return response
        return _FakeHttpResponse(response)

    def stream(
        self,
        method: str,
        url: str,
        *,
        data: dict[str, object] | None = None,
        files: list[object] | None = None,
        headers: dict[str, str],
    ) -> _FakeStreamResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "data": data,
                "files": files,
                "headers": headers,
            }
        )
        if not self._responses:
            raise AssertionError("No fake OpenAI image stream response configured.")
        response = self._responses.pop(0)
        if not isinstance(response, _FakeStreamResponse):
            raise AssertionError("Expected a fake OpenAI stream response.")
        return response


class SnapshotImageRendererArchitectureTest(unittest.TestCase):
    def test_render_config_defaults_to_gpt_image_2(self) -> None:
        config = snapshot_image_renderer.SnapshotImageRenderConfig()

        self.assertEqual(config.openai_image_model, "gpt-image-2")
        self.assertEqual(config.openai_image_size, "1536x1024")
        self.assertEqual(config.openai_image_quality, "medium")
        self.assertEqual(config.openai_image_stream_partial_images, 2)
        self.assertFalse(config.include_geometry_references)
        self.assertFalse(config.include_locked_geometry_references)
        self.assertTrue(config.include_preset_reference_images)
        self.assertFalse(config.simple_prompt_mode)

    def test_request_quality_and_size_override_openai_form_fields(self) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                image_quality="high",
                image_size="1024x1536",
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1536x1024",
                    openai_image_quality="medium",
                ),
            )

        request_call = fake_http_client.calls[0]
        self.assertEqual(request_call["data"]["quality"], "high")
        self.assertEqual(request_call["data"]["size"], "1024x1536")
        self.assertEqual(result["request"]["image_quality_requested"], "high")
        self.assertEqual(result["request"]["image_size_requested"], "1024x1536")
        self.assertFalse(result["request"]["streamed"])
        self.assertEqual(result["request"]["stream_partial_images"], 0)

    def test_streaming_form_emits_partial_and_completed_image(self) -> None:
        fake_http_client = _FakeHttpClient(
            [_FakeStreamResponse(_openai_stream_chunks())]
        )
        stream_events: list[dict[str, object]] = []
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                streamed=True,
                stream_event_handler=lambda event: stream_events.append(dict(event)),
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1536x1024",
                    openai_image_quality="medium",
                    openai_image_stream_partial_images=2,
                ),
            )

        request_call = fake_http_client.calls[0]
        self.assertEqual(request_call["method"], "POST")
        self.assertEqual(request_call["data"]["stream"], "true")
        self.assertEqual(request_call["data"]["partial_images"], "2")
        self.assertEqual(result["image"]["partial_image_count"], 1)
        self.assertEqual(result["image"]["usage"]["total_tokens"], 30)
        self.assertTrue(result["request"]["streamed"])
        self.assertEqual(result["request"]["stream_partial_images"], 2)
        self.assertEqual(stream_events[0]["type"], "status")
        partial_events = [
            event for event in stream_events if event.get("type") == "partial_image"
        ]
        self.assertEqual(len(partial_events), 1)
        self.assertEqual(partial_events[0]["partial_image_index"], 1)
        self.assertTrue(str(partial_events[0]["image_data_url"]).startswith("data:"))

    def test_openai_generate_ignores_geometry_references_by_default(self) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                locked_geometry_atlases=[
                    snapshot_image_renderer.SnapshotLockedGeometryAtlas(
                        atlas_index=1,
                        image_data_url=_tiny_png_data_url(),
                        source="clean_3d_visible_object_atlas",
                        objects=[],
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                ),
            )

        files = fake_http_client.calls[0]["files"]
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(image_file_names, ["source.png"])
        self.assertEqual(
            result["request"]["geometry_references_requested"],
            {
                "depth_map": True,
                "edge_map": True,
                "segmentation_map": True,
            },
        )
        self.assertEqual(
            result["request"]["geometry_references_used"],
            {
                "depth_map": False,
                "edge_map": False,
                "segmentation_map": False,
            },
        )
        self.assertFalse(result["request"]["include_geometry_references_config"])
        self.assertEqual(result["request"]["locked_geometry_atlas_requested_count"], 1)
        self.assertEqual(result["request"]["locked_geometry_atlas_used_count"], 0)

    def test_simple_prompt_mode_sends_source_and_preset_references(self) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                depth_map_image_data_url=_tiny_png_data_url(),
                edge_map_image_data_url=_tiny_png_data_url(),
                segmentation_map_image_data_url=_tiny_png_data_url(),
                preset_selection={
                    "style": "japandi",
                    "lighting": "daylight_4",
                    "scenery": "old_town",
                },
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    simple_prompt_mode=True,
                    strict_layout_mode=False,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
                "preset_style_1.png",
                "preset_lighting_2.png",
                "preset_scenery_3.png",
            ],
        )
        prompt = fake_http_client.calls[0]["data"]["prompt"]
        self.assertTrue(prompt.startswith("Image 1 is the only target"))
        self.assertIn(
            "Images 2-4 are preset references only",
            prompt,
        )
        self.assertIn(
            "Image 2: style preset reference (Japandi) only.",
            prompt,
        )
        self.assertIn(
            "Image 3: lighting preset reference (Ban ngày 4",
            prompt,
        )
        self.assertIn(
            "Image 4: scenery preset reference (Phố cổ) only.",
            prompt,
        )
        self.assertIn(
            "Do not copy room, camera, furniture, openings, layout, object count",
            prompt,
        )
        self.assertIn(
            "make this more photorealistic, don't change anything.",
            prompt,
        )
        self.assertIn("Style preset (Japandi)", prompt)
        self.assertIn("Lighting preset (Ban ngày 4", prompt)
        self.assertIn("Scenery preset (Phố cổ)", prompt)
        self.assertNotIn("You are converting", prompt)
        self.assertNotIn("Layout lock instructions", prompt)
        self.assertTrue(result["request"]["simple_prompt_used"])
        self.assertTrue(result["request"]["simple_prompt_image_role_hint_used"])
        self.assertEqual(result["request"]["preset_reference_image_count"], 3)
        self.assertEqual(
            result["request"]["geometry_references_used"],
            {
                "depth_map": False,
                "edge_map": False,
                "segmentation_map": False,
            },
        )

    def test_edit_camera_transfer_uses_render_source_and_target_3d_reference(
        self,
    ) -> None:
        fake_http_client = _FakeHttpClient([_openai_image_payload()])
        with mock.patch.object(snapshot_image_renderer, "httpx") as mocked_httpx:
            mocked_httpx.Client.return_value = fake_http_client
            result = snapshot_image_renderer.render_snapshot_image(
                _snapshot(),
                snapshot_image_data_url=_tiny_png_data_url(),
                annotated_reference_image_data_url=_tiny_png_data_url(),
                edit_source_image_data_url=_tiny_png_data_url(),
                render_mode="edit",
                scene_reference_mode="scene_reference_camera_transfer",
                preset_selection={"style": "japandi"},
                edit_operations=[
                    snapshot_image_renderer.SnapshotEditOperation(
                        object_id="R1",
                        object_name="boxed object",
                        operation_type="remove",
                        bbox_norm={
                            "x": 0.25,
                            "y": 0.25,
                            "width": 0.25,
                            "height": 0.25,
                        },
                    )
                ],
                config=snapshot_image_renderer.SnapshotImageRenderConfig(
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1536x1024",
                    openai_image_quality="medium",
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
                "camera_reference.png",
                "edit_guide.png",
                "preset_style_1.png",
            ],
        )
        self.assertFalse(any(field_name == "mask" for field_name, _ in files))
        prompt = fake_http_client.calls[0]["data"]["prompt"]
        self.assertIn("transferring it to a new 3D camera view", prompt)
        self.assertIn("Do not preserve the old camera", prompt)
        self.assertIn("Transform Image 1", prompt)
        self.assertIn("Image 2 is the sharp target 3D layout/camera reference", prompt)
        self.assertIn("Image 3 is a structural geometry guide", prompt)
        self.assertIn("Image 4 is a changed preset reference only", prompt)
        self.assertIn("Remove the object or visual content", prompt)
        self.assertNotIn("Preserve room identity, materials, style continuity", prompt)
        self.assertTrue(result["request"]["camera_transfer_reference_used"])
        self.assertTrue(result["request"]["edit_region_guide_used"])
        self.assertTrue(result["request"]["annotated_reference_used"])
        self.assertEqual(result["request"]["preset_reference_image_count"], 1)
        self.assertEqual(
            result["request"]["scene_reference_mode"],
            "scene_reference_camera_transfer",
        )

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
                    include_geometry_references=True,
                    include_preset_reference_images=False,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
        self.assertTrue(result["request"]["annotated_reference_requested"])
        self.assertFalse(result["request"]["annotated_reference_used"])
        self.assertFalse(result["request"]["include_annotated_reference_config"])
        self.assertGreaterEqual(len(result["request"]["locked_items"]), 2)

        request_call = fake_http_client.calls[0]
        self.assertEqual(request_call["url"], "https://api.openai.com/v1/images/edits")
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
            ],
        )
        prompt = request_call["data"]["prompt"]
        self.assertIn("Image 1 is the clean 3D source render", prompt)
        self.assertIn("Image 2 is a deterministic depth map", prompt)
        self.assertIn("Image 3 is a deterministic edge map", prompt)
        self.assertIn("Image 4 is a deterministic segmentation map", prompt)
        self.assertNotIn("same-camera labeled structural guide", prompt)
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
                    include_geometry_references=True,
                    include_locked_geometry_references=True,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
                    include_geometry_references=True,
                    include_locked_geometry_references=True,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
                    include_geometry_references=True,
                    include_locked_geometry_references=True,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
                    include_geometry_references=True,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
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
        self.assertEqual(image_file_names, ["source.png"])
        self.assertTrue(result["request"]["annotated_reference_requested"])
        self.assertFalse(result["request"]["annotated_reference_used"])

    def test_openai_generate_can_include_annotated_reference_when_enabled(
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
                    include_annotated_reference=True,
                    openai_api_key="secret",
                    openai_image_model="gpt-image-2",
                    openai_image_size="1024x1024",
                    openai_image_quality="medium",
                ),
            )

        files = fake_http_client.calls[0]["files"]
        image_file_names = [
            payload[0] for field_name, payload in files if field_name == "image[]"
        ]
        self.assertEqual(image_file_names, ["source.png", "edit_guide.png"])
        self.assertTrue(result["request"]["annotated_reference_requested"])
        self.assertTrue(result["request"]["annotated_reference_used"])
        self.assertTrue(result["request"]["include_annotated_reference_config"])
        prompt = fake_http_client.calls[0]["data"]["prompt"]
        self.assertIn("Image 2 is a same-camera labeled structural guide", prompt)

    def test_render_config_reads_annotated_reference_env_toggle(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TKNT_SNAPSHOT_RENDER_INCLUDE_ANNOTATED_REFERENCE": "true",
                "TKNT_SNAPSHOT_RENDER_INCLUDE_GEOMETRY_REFERENCES": "true",
                "TKNT_SNAPSHOT_RENDER_INCLUDE_LOCKED_GEOMETRY_REFERENCES": "true",
                "TKNT_SNAPSHOT_RENDER_INCLUDE_PRESET_REFERENCE_IMAGES": "false",
                "TKNT_SNAPSHOT_RENDER_SIMPLE_PROMPT": "true",
            },
            clear=True,
        ):
            config = snapshot_image_renderer.SnapshotImageRenderConfig.from_env()

        self.assertTrue(config.include_annotated_reference)
        self.assertTrue(config.include_geometry_references)
        self.assertTrue(config.include_locked_geometry_references)
        self.assertFalse(config.include_preset_reference_images)
        self.assertTrue(config.simple_prompt_mode)


if __name__ == "__main__":
    unittest.main()
