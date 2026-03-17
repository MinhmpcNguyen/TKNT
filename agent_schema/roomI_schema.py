from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: int
    y: int


class RoomModel(BaseModel):
    unit: Literal["mm"]
    room_id: str
    name: str
    polygon_ccw: list[Point] = Field(default_factory=list)


class DoorOpening(BaseModel):
    id: str
    segment_mm: list[Point]
    swing_radius_mm: int
    hinge_hint: Literal["UNKNOWN", "LEFT", "RIGHT"]

    @field_validator("segment_mm")
    @classmethod
    def _validate_segment(cls, value: list[Point]) -> list[Point]:
        if len(value) != 2:
            raise ValueError("segment_mm must have exactly 2 points")
        return value


class WindowOpening(BaseModel):
    id: str
    segment_mm: list[Point]
    clearance_mm: int

    @field_validator("segment_mm")
    @classmethod
    def _validate_segment(cls, value: list[Point]) -> list[Point]:
        if len(value) != 2:
            raise ValueError("segment_mm must have exactly 2 points")
        return value


class Openings(BaseModel):
    doors: list[DoorOpening] = Field(default_factory=list)
    windows: list[WindowOpening] = Field(default_factory=list)


class Obstacle(BaseModel):
    id: str
    type: Literal["door_swing", "no_go", "window_clearance"]
    polygon_ccw: list[Point] = Field(default_factory=list)
    hard: bool


class Meta(BaseModel):
    room_type: str
    style: str
    height_mm: int
    window_direction: str
    max_iterations: int


class RoomInterpreterOutput(BaseModel):
    status: Literal["OK", "NEED_INFO", "CONFLICT"]
    room: RoomModel
    openings: Openings
    obstacles: list[Obstacle] = Field(default_factory=list)
    meta: Meta
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
