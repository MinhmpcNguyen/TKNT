from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AllowedRotation = Literal[0, 90, 180, 270]
PlacerStatus = Literal["OK", "UNSAT", "NEED_INFO"]


def _strip_required_str(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ClusterTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    x: int
    y: int
    rot: AllowedRotation

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")


class MacroClusterPlacerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PlacerStatus
    grid_mm: int | None = None
    cluster_transforms: list[ClusterTransform] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("grid_mm")
    @classmethod
    def _grid_mm_positive_if_present(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("grid_mm must be > 0")
        return value

    @field_validator("notes")
    @classmethod
    def _notes_non_empty(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, "notes item"))
        return out

    @field_validator("cluster_transforms")
    @classmethod
    def _unique_cluster_ids(
        cls, value: list[ClusterTransform]
    ) -> list[ClusterTransform]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_transforms cluster_id must be unique")
        return value

    @model_validator(mode="after")
    def _require_fields_for_ok(self) -> "MacroClusterPlacerOutput":
        if self.status == "OK":
            if self.grid_mm is None:
                raise ValueError("grid_mm is required when status=OK")
            if self.grid_mm <= 0:
                raise ValueError("grid_mm must be > 0 when status=OK")
            if not self.cluster_transforms:
                raise ValueError("cluster_transforms required when status=OK")
        return self
