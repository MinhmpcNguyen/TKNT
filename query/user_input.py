from __future__ import annotations

import logging
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

RoomType = Literal[
    "bedroom",
    "living_room",
    "kitchen",
    "bathroom",
    "office",
    "dining_room",
    "other",
]

StyleType = Literal[
    "modern",
    "classic",
    "minimal",
    "industrial",
    "scandinavian",
    "japanese",
    "other",
]

WindowDirection = Literal[
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "unknown",
]


class Point(BaseModel):
    x: float
    y: float


class ShapePoint(Point):
    kind: Literal["wall", "door"] = "wall"
    door_width_mm: float | None = None
    hinge_side: Literal["left", "right"] | None = None
    open_direction: Literal["in", "out"] | None = None


class UserInput(BaseModel):
    description: str = Field(min_length=1)
    room_type: str | None = None
    floor_area_m2: float | None = None
    height: float | None = None
    shape_points: list[ShapePoint] | None = None
    location_floor: int | None = None
    bathrooms: int | None = None
    windows: int | None = None
    window_direction: str | None = None
    style: str | None = None
    special_description: str | None = None
    feng_shui: str | None = None

    @field_validator(
        "description",
        "special_description",
        "feng_shui",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else None
        return value

    @field_validator("floor_area_m2", mode="before")
    @classmethod
    def _coerce_area(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float, str)):
            return float(value)
        return value

    @field_validator("bathrooms", "windows", "location_floor", mode="before")
    @classmethod
    def _coerce_int(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float, str)):
            return int(value)
        return value

    @model_validator(mode="after")
    def _validate_non_negative(self) -> "UserInput":
        if self.floor_area_m2 is not None and self.floor_area_m2 <= 0:
            raise ValueError("floor_area_m2 must be positive.")
        if self.bathrooms is not None and self.bathrooms < 0:
            raise ValueError("bathrooms must be >= 0.")
        if self.windows is not None and self.windows < 0:
            raise ValueError("windows must be >= 0.")
        return self

    def normalize_room_type(self) -> RoomType:
        if not self.room_type:
            return "other"
        normalized = _normalize_text(self.room_type)
        mapping: dict[str, RoomType] = {
            "bedroom": "bedroom",
            "phong ngu": "bedroom",
            "living room": "living_room",
            "living_room": "living_room",
            "phong khach": "living_room",
            "kitchen": "kitchen",
            "phong bep": "kitchen",
            "bathroom": "bathroom",
            "phong tam": "bathroom",
            "office": "office",
            "phong lam viec": "office",
            "dining room": "dining_room",
            "dining_room": "dining_room",
            "phong an": "dining_room",
        }
        return mapping.get(normalized, "other")

    def normalize_style(self) -> StyleType:
        if not self.style:
            return "other"
        normalized = _normalize_text(self.style)
        mapping: dict[str, StyleType] = {
            "modern": "modern",
            "hien dai": "modern",
            "classic": "classic",
            "co dien": "classic",
            "minimal": "minimal",
            "toi gian": "minimal",
            "industrial": "industrial",
            "scandinavian": "scandinavian",
            "japanese": "japanese",
            "nhat ban": "japanese",
        }
        return mapping.get(normalized, "other")

    def normalize_window_direction(self) -> WindowDirection:
        if not self.window_direction:
            return "unknown"
        normalized = _normalize_text(self.window_direction)
        mapping: dict[str, WindowDirection] = {
            "north": "north",
            "south": "south",
            "east": "east",
            "west": "west",
            "northeast": "northeast",
            "northwest": "northwest",
            "southeast": "southeast",
            "southwest": "southwest",
            "bac": "north",
            "nam": "south",
            "dong": "east",
            "tay": "west",
            "dong bac": "northeast",
            "dong nam": "southeast",
            "tay bac": "northwest",
            "tay nam": "southwest",
        }
        return mapping.get(normalized, "unknown")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_diacritics = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return without_diacritics.strip().lower()


class UserInputParser:
    def __init__(self) -> None:
        self._last_error: ValidationError | None = None

    @property
    def last_error(self) -> ValidationError | None:
        return self._last_error

    def parse(self, payload: Mapping[str, object]) -> UserInput:
        try:
            self._last_error = None
            return UserInput.model_validate(payload)
        except ValidationError as exc:
            self._last_error = exc
            logger.exception("Invalid user input payload.")
            raise

    def parse_from_pairs(self, pairs: Sequence[tuple[str, object]]) -> UserInput:
        payload = dict(pairs)
        return self.parse(payload)
