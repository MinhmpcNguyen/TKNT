from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypeAlias, cast

from PIL import Image, ImageDraw

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
EditOperationType = Literal["prompt", "add", "replace", "recolor", "remove"]

_OBJECT_SCOPE_MASK_OPERATION_TYPES = {"add", "replace"}
_OBJECT_SCOPE_MASK_PADDING_RATIO = 0.012
_OBJECT_SCOPE_MASK_NEAR_PADDING_RATIO = 0.03
_OBJECT_SCOPE_MIN_EDITABLE_PIXEL_RATIO = 0.002
_OBJECT_SCOPE_MAX_LOCKED_TARGET_FRACTION = 0.75
_MAX_REASONABLE_OBJECT_AREA_RATIO = 0.65


class ObjectScopeEditOperation(Protocol):
    object_id: str
    object_name: str | None
    operation_type: EditOperationType | None
    bbox_norm: Mapping[str, float] | None


def should_use_object_scope_edit_mask(
    edit_operations: Sequence[ObjectScopeEditOperation],
) -> bool:
    return any(
        (operation.operation_type or "prompt") in _OBJECT_SCOPE_MASK_OPERATION_TYPES
        for operation in edit_operations
    )


def object_scope_edit_mask_image(
    *,
    image_size: tuple[int, int],
    source_payload: Mapping[str, object],
    edit_operations: Sequence[ObjectScopeEditOperation],
) -> tuple[Image.Image, dict[str, JSONValue]]:
    image_width, image_height = image_size
    mask = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    padding_px = max(
        1, round(min(image_width, image_height) * _OBJECT_SCOPE_MASK_PADDING_RATIO)
    )
    near_padding_px = max(
        padding_px,
        round(min(image_width, image_height) * _OBJECT_SCOPE_MASK_NEAR_PADDING_RATIO),
    )
    object_candidates = _visible_object_candidates(
        source_payload,
        image_size=image_size,
    )
    operation_records: list[dict[str, JSONValue]] = []
    protected_objects: list[dict[str, JSONValue]] = []
    editable_region_count = 0

    for index, operation in enumerate(edit_operations, start=1):
        if operation.bbox_norm is None:
            continue
        operation_bbox = _bbox_norm_to_screen_bbox(
            operation.bbox_norm,
            image_size=image_size,
        )
        if operation_bbox is None:
            continue

        _draw_editable_bbox(
            draw,
            operation_bbox,
            padding_px=padding_px,
            image_size=image_size,
        )
        editable_region_count += 1
        target_tokens = _target_tokens(operation)
        operation_type = operation.operation_type or "prompt"
        expanded_bbox = _expand_screen_bbox(
            operation_bbox,
            padding_px=near_padding_px,
            image_size=image_size,
        )
        protected_for_operation: list[dict[str, JSONValue]] = []

        for candidate in object_candidates:
            candidate_bbox = _float_mapping(candidate.get("screen_bbox_px"))
            if not candidate_bbox:
                continue
            is_target = (
                operation_type != "add"
                and bool(target_tokens)
                and _candidate_matches_target(candidate, target_tokens)
            )
            if is_target or not _bbox_should_be_preserved(
                candidate_bbox, expanded_bbox
            ):
                continue
            rejection_reason = _protection_rejection_reason(
                candidate=candidate,
                candidate_bbox=candidate_bbox,
                operation_bbox=operation_bbox,
                image_size=image_size,
                padding_px=padding_px,
            )
            if rejection_reason is not None:
                continue

            _draw_preserved_bbox(
                draw,
                candidate_bbox,
                padding_px=padding_px,
                image_size=image_size,
            )
            protected = {
                "operation_id": operation.object_id,
                "object_id": _json_safe(candidate.get("object_id")),
                "object_type": _json_safe(candidate.get("object_type")),
                "reason": "non_target_object_overlaps_edit_region",
                "screen_bbox_px": _json_safe(candidate_bbox),
                "bbox_norm": _json_safe(candidate.get("bbox_norm")),
            }
            protected_for_operation.append(protected)
            protected_objects.append(protected)

        operation_records.append(
            {
                "operation_index": index,
                "operation_id": operation.object_id,
                "operation_type": operation_type,
                "object_name": operation.object_name,
                "target_tokens": target_tokens,
                "screen_bbox_px": _json_safe(operation_bbox),
                "protected_object_count": len(protected_for_operation),
                "protected_objects": protected_for_operation[:20],
            }
        )

    if editable_region_count == 0:
        return mask, {
            "enabled": False,
            "image": None,
            "reason": "no_bbox_edit_operations",
            "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
            "editable_region_count": 0,
            "protected_object_count": 0,
            "operations": operation_records,
        }

    editable_ratio = _mask_editable_pixel_ratio(mask)
    if editable_ratio < _OBJECT_SCOPE_MIN_EDITABLE_PIXEL_RATIO and protected_objects:
        fallback_mask, fallback_metadata = _bbox_only_mask_image(
            image_size=image_size,
            edit_operations=edit_operations,
            padding_px=padding_px,
        )
        return fallback_mask, {
            **fallback_metadata,
            "fallback_reason": "protected_objects_closed_edit_region",
            "discarded_protected_object_count": len(protected_objects),
            "discarded_protected_objects": protected_objects[:80],
            "operations": operation_records,
        }

    return mask, {
        "enabled": True,
        "image": None,
        "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
        "editable_region_count": editable_region_count,
        "editable_pixel_ratio": round(editable_ratio, 6),
        "preserved_pixel_ratio": round(1.0 - editable_ratio, 6),
        "protected_object_count": len(protected_objects),
        "protected_objects": protected_objects[:80],
        "operations": operation_records,
    }


def _visible_object_candidates(
    source_payload: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> list[dict[str, JSONValue]]:
    visible_objects = source_payload.get("visibleObjects")
    if not isinstance(visible_objects, list):
        return []
    candidates: list[dict[str, JSONValue]] = []
    for item in visible_objects:
        payload = _mapping(item)
        object_id = _text(payload.get("id"))
        bbox = _screen_bbox(payload.get("screenBboxPx"))
        if object_id is None or bbox is None:
            continue
        normalized = _normalized_bbox_metrics(
            bbox,
            image_width=image_size[0],
            image_height=image_size[1],
        )
        object_type = _object_type_label(payload)
        object_name = (
            _text(payload.get("label"))
            or _text(payload.get("name"))
            or _text(payload.get("type"))
            or object_id
        )
        candidates.append(
            {
                "object_id": object_id,
                "object_type": object_type,
                "object_name": object_name,
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
        )
    return candidates


def _protection_rejection_reason(
    *,
    candidate: Mapping[str, object],
    candidate_bbox: Mapping[str, float],
    operation_bbox: Mapping[str, float],
    image_size: tuple[int, int],
    padding_px: int,
) -> str | None:
    area_ratio = _float_value(candidate.get("area_ratio")) or 0.0
    if area_ratio > _MAX_REASONABLE_OBJECT_AREA_RATIO:
        return "object_bbox_too_large"

    protected_bbox = _expand_screen_bbox(
        candidate_bbox,
        padding_px=padding_px,
        image_size=image_size,
    )
    operation_area = max(1.0, _bbox_area(operation_bbox))
    locked_fraction = _bbox_intersection_area(protected_bbox, operation_bbox) / (
        operation_area
    )
    if locked_fraction > _OBJECT_SCOPE_MAX_LOCKED_TARGET_FRACTION:
        return "would_close_edit_region"
    return None


def _bbox_only_mask_image(
    *,
    image_size: tuple[int, int],
    edit_operations: Sequence[ObjectScopeEditOperation],
    padding_px: int,
) -> tuple[Image.Image, dict[str, JSONValue]]:
    mask = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    editable_region_count = 0
    for operation in edit_operations:
        if operation.bbox_norm is None:
            continue
        operation_bbox = _bbox_norm_to_screen_bbox(
            operation.bbox_norm,
            image_size=image_size,
        )
        if operation_bbox is None:
            continue
        _draw_editable_bbox(
            draw,
            operation_bbox,
            padding_px=padding_px,
            image_size=image_size,
        )
        editable_region_count += 1

    editable_ratio = _mask_editable_pixel_ratio(mask)
    return mask, {
        "enabled": editable_region_count > 0,
        "image": None,
        "mask_semantics": "transparent_pixels_are_editable_opaque_pixels_are_preserved",
        "editable_region_count": editable_region_count,
        "editable_pixel_ratio": round(editable_ratio, 6),
        "preserved_pixel_ratio": round(1.0 - editable_ratio, 6),
        "protected_object_count": 0,
        "protected_objects": [],
    }


def _target_tokens(operation: ObjectScopeEditOperation) -> list[str]:
    tokens: list[str] = []
    for raw_token in (operation.object_id, operation.object_name):
        normalized = _normalized_token(raw_token)
        if normalized and normalized not in tokens and normalized != "manual_box":
            tokens.append(normalized)
    return tokens


def _candidate_matches_target(
    candidate: Mapping[str, object],
    target_tokens: Sequence[str],
) -> bool:
    candidate_tokens = [
        _normalized_token(candidate.get("object_id")),
        _normalized_token(candidate.get("object_type")),
        _normalized_token(candidate.get("object_name")),
    ]
    for target_token in target_tokens:
        for candidate_token in candidate_tokens:
            if not candidate_token:
                continue
            if target_token == candidate_token:
                return True
            if len(target_token) >= 4 and target_token in candidate_token:
                return True
            if len(candidate_token) >= 4 and candidate_token in target_token:
                return True
    return False


def _bbox_should_be_preserved(
    candidate_bbox: Mapping[str, float],
    edit_bbox: Mapping[str, float],
) -> bool:
    if _bbox_intersection_area(candidate_bbox, edit_bbox) > 0:
        return True
    center_x = candidate_bbox["minX"] + candidate_bbox["width"] / 2
    center_y = candidate_bbox["minY"] + candidate_bbox["height"] / 2
    return (
        edit_bbox["minX"] <= center_x <= edit_bbox["maxX"]
        and edit_bbox["minY"] <= center_y <= edit_bbox["maxY"]
    )


def _bbox_norm_to_screen_bbox(
    bbox_norm: Mapping[str, float],
    *,
    image_size: tuple[int, int],
) -> dict[str, float] | None:
    image_width, image_height = image_size
    x = _float_value(bbox_norm.get("x"))
    y = _float_value(bbox_norm.get("y"))
    width = _float_value(bbox_norm.get("width"))
    height = _float_value(bbox_norm.get("height"))
    if x is None or y is None or width is None or height is None:
        return None
    min_x = max(0.0, min(float(image_width), x * image_width))
    min_y = max(0.0, min(float(image_height), y * image_height))
    max_x = max(0.0, min(float(image_width), (x + width) * image_width))
    max_y = max(0.0, min(float(image_height), (y + height) * image_height))
    if max_x <= min_x or max_y <= min_y:
        return None
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def _screen_bbox(value: object) -> dict[str, float] | None:
    payload = _mapping(value)
    min_x = _float_value(payload.get("minX"))
    min_y = _float_value(payload.get("minY"))
    if min_x is None:
        min_x = _float_value(payload.get("x"))
    if min_y is None:
        min_y = _float_value(payload.get("y"))
    width = _float_value(payload.get("width"))
    height = _float_value(payload.get("height"))
    max_x = _float_value(payload.get("maxX"))
    max_y = _float_value(payload.get("maxY"))
    if min_x is None or min_y is None:
        return None
    if max_x is None and width is not None:
        max_x = min_x + width
    if max_y is None and height is not None:
        max_y = min_y + height
    if max_x is None or max_y is None:
        return None
    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)
    if width <= 0.0 or height <= 0.0:
        return None
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": width,
        "height": height,
    }


def _expand_screen_bbox(
    bbox: Mapping[str, float],
    *,
    padding_px: int,
    image_size: tuple[int, int],
) -> dict[str, float]:
    image_width, image_height = image_size
    min_x = max(0.0, bbox["minX"] - padding_px)
    min_y = max(0.0, bbox["minY"] - padding_px)
    max_x = min(float(image_width), bbox["maxX"] + padding_px)
    max_y = min(float(image_height), bbox["maxY"] + padding_px)
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max(0.0, max_x - min_x),
        "height": max(0.0, max_y - min_y),
    }


def _draw_editable_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    padding_px: int,
    image_size: tuple[int, int],
) -> None:
    expanded = _expand_screen_bbox(bbox, padding_px=padding_px, image_size=image_size)
    draw.rectangle(
        [expanded["minX"], expanded["minY"], expanded["maxX"], expanded["maxY"]],
        fill=(0, 0, 0, 0),
    )


def _draw_preserved_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Mapping[str, float],
    *,
    padding_px: int,
    image_size: tuple[int, int],
) -> None:
    expanded = _expand_screen_bbox(bbox, padding_px=padding_px, image_size=image_size)
    draw.rectangle(
        [expanded["minX"], expanded["minY"], expanded["maxX"], expanded["maxY"]],
        fill=(255, 255, 255, 255),
    )


def _mask_editable_pixel_ratio(mask: Image.Image) -> float:
    alpha_values = list(mask.getchannel("A").getdata())
    editable_pixels = sum(1 for value in alpha_values if value == 0)
    return editable_pixels / max(1, len(alpha_values))


def _normalized_bbox_metrics(
    bbox: Mapping[str, float],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    width = max(0.0, bbox["width"])
    height = max(0.0, bbox["height"])
    area = (width * height) / max(1.0, float(image_width * image_height))
    edge_touch_count = float(
        int(bbox["minX"] <= 1.0)
        + int(bbox["minY"] <= 1.0)
        + int(bbox["maxX"] >= image_width - 1)
        + int(bbox["maxY"] >= image_height - 1)
    )
    return {
        "x": bbox["minX"] / max(1.0, float(image_width)),
        "y": bbox["minY"] / max(1.0, float(image_height)),
        "width": width / max(1.0, float(image_width)),
        "height": height / max(1.0, float(image_height)),
        "area": area,
        "edge_touch_count": edge_touch_count,
        "aspect_ratio": width / max(1.0, height),
    }


def _bbox_area(bbox: Mapping[str, float]) -> float:
    return max(0.0, bbox["width"]) * max(0.0, bbox["height"])


def _bbox_intersection_area(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    left = max(first["minX"], second["minX"])
    top = max(first["minY"], second["minY"])
    right = min(first["maxX"], second["maxX"])
    bottom = min(first["maxY"], second["maxY"])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _object_type_label(payload: Mapping[str, object]) -> str:
    raw_type = _text(payload.get("type")) or _text(payload.get("id")) or ""
    return re.sub(r"_\\d+$", "", _normalized_token(raw_type))


def _normalized_token(value: object) -> str:
    text = _text(value)
    if text is None:
        return ""
    normalized_chars = [
        character.lower() if character.isalnum() else "_" for character in text.strip()
    ]
    normalized = "_".join(part for part in "".join(normalized_chars).split("_") if part)
    parts = normalized.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def _float_mapping(value: object) -> Mapping[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, item in value.items():
        parsed_value = _float_value(item)
        if parsed_value is not None:
            parsed[str(key)] = parsed_value
    return parsed


def _float_value(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, dict) else {}


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
