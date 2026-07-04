from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, UnidentifiedImageError

Vec3 = tuple[float, float, float]
Point2 = tuple[float, float]

_ROOM_SHELL_DEFAULT_CEILING_HEIGHT_MM = 2700.0
_ROOM_SHELL_PROJECTION_NEAR_M = 0.05
_ROOM_SHELL_PROJECTED_POINT_LIMIT = 100000.0
_ROOM_SHELL_TOUCH_CROP_MARGIN_RATIO = 0.025
_ROOM_SHELL_NEAR_CROP_MARGIN_RATIO = 0.12
_LOCKED_GEOMETRY_MIN_BBOX_PX = 4
_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS = 16
_LAYOUT_GUARD_DEFAULT_MAX_OBJECT_AREA = 0.22
_LAYOUT_GUARD_MAX_OBJECT_AREA_BY_TYPE: Mapping[str, float] = {
    "armchair": 0.2,
    "bed": 0.34,
    "bookshelf": 0.28,
    "cabinet": 0.3,
    "coffee_table": 0.16,
    "floor_lamp": 0.08,
    "media_shelf": 0.18,
    "nightstand": 0.12,
    "rug": 0.42,
    "side_table": 0.14,
    "sofa": 0.34,
    "speaker": 0.08,
    "table_lamp": 0.08,
    "tv_console": 0.22,
    "vase": 0.08,
    "wardrobe": 0.34,
}
_LAYOUT_GUARD_SLENDER_OBJECT_TYPES = {
    "bookshelf",
    "curtain",
    "floor_lamp",
    "table_lamp",
    "wall_art",
}


@dataclass(frozen=True)
class ProjectedRoomShellSurface:
    surface_id: str
    surface_type: str
    polygon_px: list[Point2]
    average_depth_m: float
    touches_crop_edges: tuple[str, ...]
    near_crop_edges: tuple[str, ...]
    bbox_norm: dict[str, float]


@dataclass(frozen=True)
class SemanticLayoutGuide:
    image_data_url: str | None
    prompt: str | None
    metadata: dict[str, object]


def semantic_layout_lock_prompt_for_snapshot(
    snapshot_payload: Mapping[str, object],
) -> str | None:
    image_size = _snapshot_image_size(snapshot_payload)
    if image_size is None:
        return None
    surfaces, _ = _project_room_shell_surfaces(
        snapshot_payload,
        image_size=image_size,
    )
    _, rejected_objects = _layout_guard_object_candidates(
        snapshot_payload,
        image_size=image_size,
    )
    lines = _semantic_room_shell_lock_lines(
        surfaces=surfaces,
        opening_boxes=_room_opening_guide_boxes(snapshot_payload),
        rejected_objects=rejected_objects,
    )
    return _semantic_layout_lock_prompt(lines)


def semantic_layout_guide_for_snapshot_image(
    *,
    snapshot_payload: Mapping[str, object],
    source_image_data_url: str,
) -> SemanticLayoutGuide:
    source_image = _image_from_data_url(source_image_data_url).convert("RGBA")
    clean_objects, rejected_objects = _layout_guard_object_candidates(
        snapshot_payload,
        image_size=source_image.size,
    )
    opening_boxes = _room_opening_guide_boxes(snapshot_payload)
    guide, semantic_metadata = _room_shell_semantic_guide_image(
        source_image=source_image,
        source_payload=snapshot_payload,
        clean_objects=clean_objects,
        rejected_objects=rejected_objects,
        opening_boxes=opening_boxes,
    )
    lines = [
        str(item).strip()
        for item in _sequence(semantic_metadata.get("semantic_room_shell_locks"))
        if isinstance(item, str) and item.strip()
    ]
    metadata: dict[str, object] = {
        "enabled": True,
        "image": "inline_data_url",
        "guide_type": "room_shell_semantic",
        "image_size": [source_image.width, source_image.height],
        "clean_object_count": len(clean_objects),
        "rejected_object_count": len(rejected_objects),
        "opening_bbox_count": len(opening_boxes),
        "clean_objects": clean_objects,
        "rejected_objects": rejected_objects,
        **semantic_metadata,
    }
    return SemanticLayoutGuide(
        image_data_url=_png_data_url_from_image(guide),
        prompt=_semantic_layout_lock_prompt(lines),
        metadata=metadata,
    )


def _semantic_room_shell_lock_lines(
    *,
    surfaces: list[ProjectedRoomShellSurface],
    opening_boxes: list[tuple[str, dict[str, float]]],
    rejected_objects: list[dict[str, object]],
) -> list[str]:
    lines: list[str] = []
    wall_surfaces = [surface for surface in surfaces if surface.surface_type == "wall"]
    near_crop_walls = [
        surface
        for surface in wall_surfaces
        if surface.touches_crop_edges or surface.near_crop_edges
    ]
    for surface in near_crop_walls[:6]:
        edge_label = _edge_label(surface.touches_crop_edges or surface.near_crop_edges)
        proximity = (
            "touching the image crop"
            if surface.touches_crop_edges
            else "near the image crop"
        )
        bbox = _bbox_norm_label(surface.bbox_norm)
        lines.append(
            "Room-shell lock: preserve "
            f"{surface.surface_id} as a real wall plane {proximity} on the "
            f"{edge_label}; keep its wall seam, crop silhouette, perspective, "
            f"and occlusion footprint unchanged ({bbox}). Do not flatten it away, "
            "turn it into empty space, complete it as a nicer rectangular room, "
            "or convert it into a window, door, curtain, lamp, shelf, or decor."
        )
    if wall_surfaces:
        lines.append(
            "Shadow-vs-geometry lock: dark vertical lines, gradients, and cast "
            "shadows that lie inside colored wall planes are wall seams/shadows, "
            "not furniture. Never convert a wall shadow or corner edge into a "
            "floor lamp, pole, shelf, curtain, window, or door."
        )
    if opening_boxes:
        lines.append(
            "Opening lock: preserve exactly the existing blue opening box(es) "
            "from the semantic guide; keep their count, wall association, apparent "
            "size, and silhouette. Do not add new windows, doors, exterior views, "
            "curtains, or framed openings outside those boxes."
        )
    else:
        lines.append(
            "Opening lock: the source view has no trusted visible opening boxes "
            "in the semantic guide. Do not invent windows, doors, exterior views, "
            "curtains, or new wall cutouts."
        )
    for item in rejected_objects[:6]:
        lock_line = _rejected_object_lock_line(item)
        if lock_line:
            lines.append(lock_line)
    return lines


def _room_shell_semantic_guide_image(
    *,
    source_image: Image.Image,
    source_payload: Mapping[str, object],
    clean_objects: list[dict[str, object]],
    rejected_objects: list[dict[str, object]],
    opening_boxes: list[tuple[str, dict[str, float]]],
) -> tuple[Image.Image, dict[str, object]]:
    guide = source_image.convert("RGBA")
    image_size = guide.size
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    image_width, image_height = image_size
    stroke_width = max(3, round(min(image_width, image_height) / 300))

    surfaces, projection_metadata = _project_room_shell_surfaces(
        source_payload,
        image_size=image_size,
    )
    surface_colors: Mapping[str, tuple[int, int, int, int]] = {
        "floor": (245, 158, 11, 70),
        "ceiling": (168, 85, 247, 58),
        "wall": (6, 182, 212, 62),
    }
    surface_outline_colors: Mapping[str, tuple[int, int, int, int]] = {
        "floor": (217, 119, 6, 235),
        "ceiling": (126, 34, 206, 230),
        "wall": (8, 145, 178, 235),
    }
    wall_palette = (
        (6, 182, 212, 62),
        (59, 130, 246, 58),
        (16, 185, 129, 58),
        (236, 72, 153, 54),
        (34, 197, 94, 54),
        (99, 102, 241, 54),
    )
    wall_outline_palette = (
        (8, 145, 178, 235),
        (37, 99, 235, 230),
        (5, 150, 105, 230),
        (219, 39, 119, 225),
        (22, 163, 74, 225),
        (79, 70, 229, 225),
    )
    for index, surface in enumerate(
        sorted(surfaces, key=lambda item: item.average_depth_m, reverse=True)
    ):
        if len(surface.polygon_px) < 3:
            continue
        if surface.surface_type == "wall":
            fill = wall_palette[index % len(wall_palette)]
            outline = wall_outline_palette[index % len(wall_outline_palette)]
        else:
            fill = surface_colors.get(surface.surface_type, surface_colors["wall"])
            outline = surface_outline_colors.get(
                surface.surface_type,
                surface_outline_colors["wall"],
            )
        polygon = [(round(x), round(y)) for x, y in surface.polygon_px]
        draw.polygon(polygon, fill=fill)
        draw.line([*polygon, polygon[0]], fill=outline, width=stroke_width)

    crop_band = max(8, round(min(image_width, image_height) * 0.018))
    draw.rectangle(
        [0, 0, image_width - 1, image_height - 1],
        outline=(255, 132, 0, 235),
        width=max(2, stroke_width - 1),
    )
    for box in (
        [0, 0, crop_band, image_height],
        [image_width - crop_band, 0, image_width, image_height],
        [0, 0, image_width, crop_band],
        [0, image_height - crop_band, image_width, image_height],
    ):
        draw.rectangle(box, fill=(255, 132, 0, 18))

    for _, bbox in opening_boxes:
        _draw_semantic_bbox(
            draw,
            bbox,
            fill=(0, 112, 255, 46),
            outline=(0, 112, 255, 245),
            width=stroke_width + 1,
        )
    for item in clean_objects[:_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS]:
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if bbox:
            _draw_semantic_bbox(
                draw,
                bbox,
                fill=None,
                outline=(0, 190, 105, 235),
                width=max(2, stroke_width - 1),
            )
    for item in rejected_objects:
        bbox = _json_float_mapping(item.get("screen_bbox_px"))
        if bbox and bool(item.get("draw_rejected_bbox")):
            _draw_semantic_bbox(
                draw,
                bbox,
                fill=None,
                outline=(220, 38, 38, 220),
                width=max(2, stroke_width - 1),
            )

    guide = Image.alpha_composite(guide, overlay)
    metadata_surfaces: list[object] = []
    for surface in surfaces:
        metadata_surfaces.append(
            {
                "id": surface.surface_id,
                "type": surface.surface_type,
                "point_count": len(surface.polygon_px),
                "average_depth_m": round(surface.average_depth_m, 4),
                "touches_crop_edges": list(surface.touches_crop_edges),
                "near_crop_edges": list(surface.near_crop_edges),
                "bbox_norm": surface.bbox_norm,
            }
        )
    return guide, {
        "semantic_projection_status": projection_metadata["status"],
        "semantic_projection_reason": projection_metadata.get("reason"),
        "semantic_surface_count": len(surfaces),
        "semantic_surfaces": metadata_surfaces,
        "semantic_room_shell_locks": _semantic_room_shell_lock_lines(
            surfaces=surfaces,
            opening_boxes=opening_boxes,
            rejected_objects=rejected_objects,
        ),
        "semantic_color_key": {
            "wall_surfaces": "cyan/blue/green/pink translucent filled planes",
            "floor": "amber translucent filled plane",
            "ceiling": "purple translucent filled plane",
            "openings": "solid blue boxes",
            "clean_object_anchors": "green outlines",
            "rejected_noisy_anchors": "red outlines when drawable",
            "crop_guard": "orange border and crop bands",
        },
    }


def _semantic_layout_lock_prompt(lines: list[str]) -> str | None:
    if not lines:
        return None
    bullet_lines = "\n".join(f"- {line}" for line in lines[:12])
    return (
        "Semantic layout locks from the structural guide:\n"
        f"{bullet_lines}\n"
        "These locks are more important than photorealistic room cleanup. Preserve "
        "awkward crop walls, partial foreground occluders, wall shadows, and exact "
        "opening/object footprints even when they look unusual."
    )


def _draw_semantic_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    fill: tuple[int, int, int, int] | None,
    outline: tuple[int, int, int, int],
    width: int,
) -> None:
    box = [
        round(bbox["minX"]),
        round(bbox["minY"]),
        round(bbox["maxX"]),
        round(bbox["maxY"]),
    ]
    if fill is not None:
        draw.rectangle(box, fill=fill)
    draw.rectangle(box, outline=outline, width=width)


def _project_room_shell_surfaces(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> tuple[list[ProjectedRoomShellSurface], dict[str, object]]:
    room = _mapping(source_payload.get("room"))
    polygon = _room_polygon_points(room.get("polygonMm"))
    if len(polygon) < 3:
        return [], {"status": "skipped", "reason": "room_polygon_missing"}
    camera = _projection_camera(source_payload)
    if camera is None:
        return [], {"status": "skipped", "reason": "camera_missing_or_invalid"}

    center_x, center_y = _room_center_mm(room, polygon)
    ceiling_height_mm = _room_ceiling_height_mm(source_payload)
    surfaces: list[ProjectedRoomShellSurface] = []
    floor_points = [
        _room_point_to_world(point, height_mm=0.0, center_x=center_x, center_y=center_y)
        for point in polygon
    ]
    ceiling_points = [
        _room_point_to_world(
            point,
            height_mm=ceiling_height_mm,
            center_x=center_x,
            center_y=center_y,
        )
        for point in polygon
    ]
    floor_surface = _project_surface(
        "floor",
        "floor",
        floor_points,
        camera=camera,
        image_size=image_size,
    )
    ceiling_surface = _project_surface(
        "ceiling",
        "ceiling",
        ceiling_points,
        camera=camera,
        image_size=image_size,
    )
    if floor_surface is not None:
        surfaces.append(floor_surface)
    if ceiling_surface is not None:
        surfaces.append(ceiling_surface)

    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        wall_points = [
            _room_point_to_world(
                start,
                height_mm=0.0,
                center_x=center_x,
                center_y=center_y,
            ),
            _room_point_to_world(
                end,
                height_mm=0.0,
                center_x=center_x,
                center_y=center_y,
            ),
            _room_point_to_world(
                end,
                height_mm=ceiling_height_mm,
                center_x=center_x,
                center_y=center_y,
            ),
            _room_point_to_world(
                start,
                height_mm=ceiling_height_mm,
                center_x=center_x,
                center_y=center_y,
            ),
        ]
        surface = _project_surface(
            f"wall_{index + 1}",
            "wall",
            wall_points,
            camera=camera,
            image_size=image_size,
        )
        if surface is not None:
            surfaces.append(surface)
    return surfaces, {
        "status": "ok" if surfaces else "skipped",
        "reason": None if surfaces else "no_projected_surfaces",
        "ceiling_height_mm": round(ceiling_height_mm, 2),
        "room_center_mm": {"x": round(center_x, 2), "y": round(center_y, 2)},
    }


def _project_surface(
    surface_id: str,
    surface_type: str,
    points: list[Vec3],
    *,
    camera: Mapping[str, object],
    image_size: tuple[int, int],
) -> ProjectedRoomShellSurface | None:
    projected: list[Point2] = []
    depths: list[float] = []
    for point in points:
        projected_point = _project_world_point(
            point,
            camera=camera,
            image_size=image_size,
        )
        if projected_point is None:
            return None
        screen_point, depth = projected_point
        projected.append(screen_point)
        depths.append(depth)
    if len(projected) < 3:
        return None
    touches = _projected_polygon_crop_edges(
        projected,
        image_size=image_size,
        margin_ratio=_ROOM_SHELL_TOUCH_CROP_MARGIN_RATIO,
    )
    near_crop_edges = _projected_polygon_crop_edges(
        projected,
        image_size=image_size,
        margin_ratio=_ROOM_SHELL_NEAR_CROP_MARGIN_RATIO,
    )
    return ProjectedRoomShellSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        polygon_px=projected,
        average_depth_m=sum(depths) / max(1, len(depths)),
        touches_crop_edges=touches,
        near_crop_edges=near_crop_edges,
        bbox_norm=_projected_polygon_bbox_norm(projected, image_size=image_size),
    )


def _project_world_point(
    point: Vec3,
    *,
    camera: Mapping[str, object],
    image_size: tuple[int, int],
) -> tuple[Point2, float] | None:
    position = _vec3_tuple(camera.get("position"))
    right = _vec3_tuple(camera.get("right"))
    up = _vec3_tuple(camera.get("up"))
    forward = _vec3_tuple(camera.get("forward"))
    tan_half_fov = _optional_number(camera.get("tan_half_fov"))
    aspect = _optional_number(camera.get("aspect"))
    if (
        position is None
        or right is None
        or up is None
        or forward is None
        or tan_half_fov is None
        or aspect is None
    ):
        return None
    image_width, image_height = image_size
    relative = _vec_sub(point, position)
    depth = _vec_dot(relative, forward)
    if depth <= _ROOM_SHELL_PROJECTION_NEAR_M:
        return None
    camera_x = _vec_dot(relative, right)
    camera_y = _vec_dot(relative, up)
    ndc_x = camera_x / max(0.0001, depth * tan_half_fov * aspect)
    ndc_y = camera_y / max(0.0001, depth * tan_half_fov)
    screen_x = (ndc_x + 1.0) * 0.5 * image_width
    screen_y = (1.0 - ndc_y) * 0.5 * image_height
    screen_x = max(
        -_ROOM_SHELL_PROJECTED_POINT_LIMIT,
        min(_ROOM_SHELL_PROJECTED_POINT_LIMIT, screen_x),
    )
    screen_y = max(
        -_ROOM_SHELL_PROJECTED_POINT_LIMIT,
        min(_ROOM_SHELL_PROJECTED_POINT_LIMIT, screen_y),
    )
    return (screen_x, screen_y), depth


def _projection_camera(
    source_payload: Mapping[str, object],
) -> dict[str, object] | None:
    camera = _mapping(source_payload.get("camera"))
    position = _vec3_from_mapping(camera.get("positionM"))
    target = _vec3_from_mapping(camera.get("targetM"))
    fov_deg = _optional_number(camera.get("fovDeg"))
    aspect = _optional_number(camera.get("aspect"))
    if position is None or target is None or fov_deg is None or aspect is None:
        return None
    forward = _vec_normalize(_vec_sub(target, position))
    if forward is None:
        return None
    world_up: Vec3 = (0.0, 1.0, 0.0)
    right = _vec_normalize(_vec_cross(forward, world_up))
    if right is None:
        return None
    up = _vec_normalize(_vec_cross(right, forward))
    if up is None:
        return None
    return {
        "position": position,
        "forward": forward,
        "right": right,
        "up": up,
        "tan_half_fov": math.tan(math.radians(fov_deg) / 2.0),
        "aspect": max(0.0001, aspect),
    }


def _layout_guard_object_candidates(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    image_width, image_height = image_size
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return [], []
    clean_objects: list[dict[str, object]] = []
    rejected_objects: list[dict[str, object]] = []
    for item in visible_objects:
        payload = _mapping(item)
        object_id = _text(payload.get("id"))
        object_type = _object_type_label(payload)
        bbox = _screen_bbox(payload.get("screenBboxPx"))
        if object_id is None or bbox is None:
            continue
        normalized = _normalized_bbox_metrics(
            bbox,
            image_width=image_width,
            image_height=image_height,
        )
        reason = _layout_guard_bbox_rejection_reason(
            object_type=object_type,
            normalized=normalized,
        )
        serialized: dict[str, object] = {
            "object_id": object_id,
            "object_type": object_type,
            "screen_bbox_px": bbox,
            "bbox_norm": {
                "x": round(normalized["x"], 6),
                "y": round(normalized["y"], 6),
                "width": round(normalized["width"], 6),
                "height": round(normalized["height"], 6),
            },
            "area_ratio": round(normalized["area"], 6),
            "edge_touch_count": int(normalized["edge_touch_count"]),
            "aspect_ratio": round(normalized["aspect_ratio"], 6),
        }
        if reason is None:
            clean_objects.append(serialized)
            continue
        rejected_objects.append(
            {
                **serialized,
                "reason": reason,
                "mask_locked": normalized["area"] <= 0.12,
                "draw_rejected_bbox": normalized["area"] <= 0.35,
            }
        )
    clean_objects.sort(key=lambda item: float(item["area_ratio"]), reverse=True)
    return clean_objects[:_EDGE_SILHOUETTE_GUIDE_MAX_OBJECTS], rejected_objects


def _layout_guard_bbox_rejection_reason(
    *,
    object_type: str,
    normalized: Mapping[str, float],
) -> str | None:
    area = normalized["area"]
    width = normalized["width"]
    height = normalized["height"]
    aspect_ratio = normalized["aspect_ratio"]
    edge_touch_count = int(normalized["edge_touch_count"])
    max_area = _LAYOUT_GUARD_MAX_OBJECT_AREA_BY_TYPE.get(
        object_type,
        _LAYOUT_GUARD_DEFAULT_MAX_OBJECT_AREA,
    )
    if area > max_area:
        return "bbox_area_too_large_for_object_type"
    if width > 0.66 and height > 0.66 and object_type != "rug":
        return "bbox_covers_too_much_of_frame"
    if edge_touch_count >= 3 and object_type != "rug":
        return "bbox_touches_too_many_image_edges"
    if (
        aspect_ratio > 10.0 or aspect_ratio < 0.1
    ) and object_type not in _LAYOUT_GUARD_SLENDER_OBJECT_TYPES:
        return "bbox_aspect_ratio_implausible"
    return None


def _normalized_bbox_metrics(
    bbox: Mapping[str, float],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    width = max(0.0, bbox["width"] / max(1, image_width))
    height = max(0.0, bbox["height"] / max(1, image_height))
    x = max(0.0, min(1.0, bbox["minX"] / max(1, image_width)))
    y = max(0.0, min(1.0, bbox["minY"] / max(1, image_height)))
    edge_touch_count = float(
        sum(
            (
                x <= 0.01,
                y <= 0.01,
                x + width >= 0.99,
                y + height >= 0.99,
            )
        )
    )
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / max(height, 0.0001),
        "edge_touch_count": edge_touch_count,
    }


def _room_opening_guide_boxes(
    source_payload: Mapping[str, object],
) -> list[tuple[str, dict[str, float]]]:
    room = _mapping(source_payload.get("room"))
    openings = _mapping(room.get("openings"))
    boxes: list[tuple[str, dict[str, float]]] = []
    for opening_type in ("windows", "doors"):
        for index, item in enumerate(_sequence(openings.get(opening_type)), start=1):
            bbox = _screen_bbox(_mapping(item).get("screenBboxPx"))
            if bbox is not None:
                boxes.append((f"{opening_type[:-1]} {index}", bbox))
    return boxes


def _snapshot_image_size(
    source_payload: Mapping[str, object],
) -> tuple[int, int] | None:
    canvas = _mapping(source_payload.get("canvas"))
    width = _optional_number(canvas.get("widthPx"))
    height = _optional_number(canvas.get("heightPx"))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return int(round(width)), int(round(height))


def _room_polygon_points(value: object) -> list[Point2]:
    points: list[Point2] = []
    for item in _sequence(value):
        payload = _mapping(item)
        x = _optional_number(payload.get("x"))
        y = _optional_number(payload.get("y"))
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def _room_center_mm(room: Mapping[str, object], polygon: list[Point2]) -> Point2:
    bounds = _mapping(room.get("boundsMm"))
    min_x = _optional_number(bounds.get("minX"))
    max_x = _optional_number(bounds.get("maxX"))
    min_y = _optional_number(bounds.get("minY"))
    max_y = _optional_number(bounds.get("maxY"))
    if (
        min_x is not None
        and max_x is not None
        and min_y is not None
        and max_y is not None
    ):
        return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5


def _room_ceiling_height_mm(source_payload: Mapping[str, object]) -> float:
    room = _mapping(source_payload.get("room"))
    for key in ("ceilingHeightMm", "ceiling_height_mm", "heightMm", "height_mm"):
        value = _optional_number(room.get(key))
        if value is not None and value > 0:
            return value
    return _ROOM_SHELL_DEFAULT_CEILING_HEIGHT_MM


def _room_point_to_world(
    point: Point2,
    *,
    height_mm: float,
    center_x: float,
    center_y: float,
) -> Vec3:
    return (
        (point[0] - center_x) / 1000.0,
        height_mm / 1000.0,
        (center_y - point[1]) / 1000.0,
    )


def _projected_polygon_crop_edges(
    polygon: list[Point2],
    *,
    image_size: tuple[int, int],
    margin_ratio: float,
) -> tuple[str, ...]:
    image_width, image_height = image_size
    margin = max(8.0, min(image_width, image_height) * margin_ratio)
    edges: list[str] = []
    if any(x <= margin for x, _ in polygon):
        edges.append("left")
    if any(x >= image_width - margin for x, _ in polygon):
        edges.append("right")
    if any(y <= margin for _, y in polygon):
        edges.append("top")
    if any(y >= image_height - margin for _, y in polygon):
        edges.append("bottom")
    return tuple(edges)


def _projected_polygon_bbox_norm(
    polygon: list[Point2],
    *,
    image_size: tuple[int, int],
) -> dict[str, float]:
    image_width, image_height = image_size
    min_x = max(0.0, min((x for x, _ in polygon), default=0.0))
    min_y = max(0.0, min((y for _, y in polygon), default=0.0))
    max_x = min(float(image_width), max((x for x, _ in polygon), default=0.0))
    max_y = min(float(image_height), max((y for _, y in polygon), default=0.0))
    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)
    return {
        "x": round(min_x / max(1, image_width), 6),
        "y": round(min_y / max(1, image_height), 6),
        "width": round(width / max(1, image_width), 6),
        "height": round(height / max(1, image_height), 6),
    }


def _screen_bbox(value: object) -> dict[str, float] | None:
    payload = _mapping(value)
    min_x = _optional_number(payload.get("minX"))
    min_y = _optional_number(payload.get("minY"))
    if min_x is None:
        min_x = _optional_number(payload.get("x"))
    if min_y is None:
        min_y = _optional_number(payload.get("y"))
    if min_x is None or min_y is None:
        return None

    width = _optional_number(payload.get("width"))
    height = _optional_number(payload.get("height"))
    max_x = _optional_number(payload.get("maxX"))
    max_y = _optional_number(payload.get("maxY"))
    if max_x is None and width is not None:
        max_x = min_x + width
    if max_y is None and height is not None:
        max_y = min_y + height
    if max_x is None or max_y is None:
        return None

    width = max_x - min_x
    height = max_y - min_y
    if width < _LOCKED_GEOMETRY_MIN_BBOX_PX or height < _LOCKED_GEOMETRY_MIN_BBOX_PX:
        return None
    return {
        "minX": round(min_x, 2),
        "minY": round(min_y, 2),
        "maxX": round(max_x, 2),
        "maxY": round(max_y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def _json_float_mapping(value: object) -> dict[str, float]:
    payload = _mapping(value)
    parsed: dict[str, float] = {}
    for key in ("minX", "minY", "maxX", "maxY", "width", "height"):
        number = _optional_number(payload.get(key))
        if number is not None:
            parsed[key] = number
    return parsed


def _object_type_label(payload: Mapping[str, object]) -> str:
    raw = (_text(payload.get("type")) or _text(payload.get("id")) or "object").lower()
    parts = raw.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts) or "object"


def _rejected_object_lock_line(item: Mapping[str, object]) -> str:
    object_id = _text(item.get("object_id")) or "unknown_object"
    object_type = _text(item.get("object_type")) or "object"
    reason = _text(item.get("reason")) or "noisy_bbox"
    bbox = _mapping(item.get("bbox_norm"))
    edge_touch_count = _optional_number(item.get("edge_touch_count")) or 0.0
    area_ratio = _optional_number(item.get("area_ratio")) or 0.0
    bbox_label = _bbox_norm_label(bbox)
    if edge_touch_count >= 1:
        return (
            "Rejected-anchor lock: "
            f"{object_id} ({object_type}) has an unreliable edge/crop bbox "
            f"({reason}, area={area_ratio:.3f}, {bbox_label}), but the visible "
            "source pixels still represent a real cropped/foreground object or "
            "occluder. Preserve its visible silhouette and occlusion from Image 1; "
            "do not complete, remove, replace, expand, or move it based on the "
            "noisy bbox."
        )
    return (
        "Rejected-anchor lock: "
        f"{object_id} ({object_type}) has an unreliable bbox ({reason}, "
        f"area={area_ratio:.3f}, {bbox_label}). Treat the bbox as a warning only; "
        "preserve the source-visible object pixels and do not use the noisy bbox "
        "to add, move, resize, or replace objects."
    )


def _edge_label(edges: tuple[str, ...]) -> str:
    if not edges:
        return "image frame"
    if len(edges) == 1:
        return f"{edges[0]} edge"
    return ", ".join(f"{edge} edge" for edge in edges[:-1]) + f", and {edges[-1]} edge"


def _bbox_norm_label(bbox: Mapping[str, object]) -> str:
    x = _optional_number(bbox.get("x"))
    y = _optional_number(bbox.get("y"))
    width = _optional_number(bbox.get("width"))
    height = _optional_number(bbox.get("height"))
    if x is None or y is None or width is None or height is None:
        return "bbox unavailable"
    return f"bbox_norm x={x:.3f}, y={y:.3f}, width={width:.3f}, height={height:.3f}"


def _vec3_from_mapping(value: object) -> Vec3 | None:
    payload = _mapping(value)
    x = _optional_number(payload.get("x"))
    y = _optional_number(payload.get("y"))
    z = _optional_number(payload.get("z"))
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _vec3_tuple(value: object) -> Vec3 | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    x, y, z = value
    if not isinstance(x, (int, float)):
        return None
    if not isinstance(y, (int, float)):
        return None
    if not isinstance(z, (int, float)):
        return None
    return float(x), float(y), float(z)


def _vec_sub(left: Vec3, right: Vec3) -> Vec3:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _vec_dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vec_cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vec_normalize(vector: Vec3) -> Vec3 | None:
    length = math.sqrt(_vec_dot(vector, vector))
    if length <= 0.000001:
        return None
    return vector[0] / length, vector[1] / length, vector[2] / length


def _optional_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return [item for item in value]


def _image_from_data_url(data_url: str) -> Image.Image:
    header, separator, encoded = data_url.partition(",")
    if separator != "," or not header.startswith("data:image/"):
        raise ValueError("Expected an image data URL for semantic layout guide.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError(
            "Semantic layout guide image data URL is not valid base64."
        ) from exc
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "Unable to decode semantic layout guide source image."
        ) from exc


def _png_data_url_from_image(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
