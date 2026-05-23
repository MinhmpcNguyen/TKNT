from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["OK", "NEED_INFO", "UNSAT"]
ConceptFamily = Literal[
    "focal_axis",
    "open_center",
    "edge_weighted",
    "zoned",
    "daylight_oriented",
]
Priority = Literal["high", "medium", "low"]
CenterUsage = Literal["none", "partial", "primary", "open_reserved"]
PreferredWallSide = Literal[
    "center",
    "window_side",
    "top_wall",
    "right_wall",
    "bottom_wall",
    "left_wall",
]
CornerHint = Literal[
    "none",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "near_window",
    "far_from_entry",
]
AnchorStrength = Literal["hard", "strong", "medium"]
AnchorLockStrength = Literal["hard", "strong", "soft"]
ObjectPlacementRole = Literal[
    "anchor",
    "support",
    "functional_support",
    "paired_seat",
    "visual_focus",
    "storage",
    "decor",
]
ObjectFacingIntent = Literal[
    "front_to_open_space",
    "front_to_room_center",
    "front_to_window",
    "front_to_entry",
    "back_to_wall",
    "parallel_to_wall",
    "face_target",
    "face_away_from_target",
]
ObjectOrientationIntent = Literal[
    "front_to_open_space",
    "front_to_cluster_center",
    "front_to_room_center",
    "front_to_window",
    "front_to_entry",
    "back_to_wall",
    "side_to_wall",
    "long_axis_parallel_wall",
    "long_axis_perpendicular_wall",
    "align_with_cluster_axis",
    "face_object",
    "face_away_from_object",
    "preserve_front_access",
]
RelationType = Literal[
    "near",
    "separate",
    "adjacent_if_possible",
    "far_if_possible",
]
ViewingPairType = Literal["face_each_other", "viewer_faces_target"]


def _strip_required_str(value: str, field_name: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} must be a non-empty string")
    return clean


def _strip_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    if not clean:
        raise ValueError("optional string fields must be null or non-empty strings")
    return clean


class VisualClusterZoneDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    zone_assignment: str
    preferred_wall_side: PreferredWallSide = "center"
    center_usage: CenterUsage = "none"
    anchor_strength: AnchorStrength = "medium"
    priority: Priority = "medium"
    reason: str

    @field_validator("cluster_id", "zone_assignment", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)


class VisualObjectOrientationDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    object_id: str
    intents: list[ObjectOrientationIntent] = Field(default_factory=list)
    target_object_id: str | None = None
    target_object_cluster_id: str | None = None
    priority: Priority = "medium"
    reason: str

    @field_validator("cluster_id", "object_id", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("target_object_id", "target_object_cluster_id")
    @classmethod
    def _optional_non_empty_str(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @field_validator("intents")
    @classmethod
    def _unique_intents(
        cls, value: list[ObjectOrientationIntent]
    ) -> list[ObjectOrientationIntent]:
        if len(value) != len(set(value)):
            raise ValueError("intents must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_target_logic(self) -> "VisualObjectOrientationDirective":
        targets_object = any(
            intent in {"face_object", "face_away_from_object"}
            for intent in self.intents
        )
        if targets_object and self.target_object_id is None:
            raise ValueError(
                "target_object_id is required for face_object/face_away_from_object"
            )
        if not targets_object and (
            self.target_object_id is not None
            or self.target_object_cluster_id is not None
        ):
            raise ValueError(
                "target object fields are only allowed for object-facing intents"
            )
        if (
            self.target_object_id is not None
            and self.target_object_id == self.object_id
            and self.target_object_cluster_id in {None, self.cluster_id}
        ):
            raise ValueError("target object must differ from source object")
        return self


class VisualObjectRegionDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    object_id: str
    region_id: str
    wall_side: PreferredWallSide = "center"
    corner_hint: CornerHint = "none"
    semantic_anchor_region_id: str | None = None
    anchor_lock_strength: AnchorLockStrength = "hard"
    placement_role: ObjectPlacementRole = "support"
    facing_intent: ObjectFacingIntent = "front_to_open_space"
    target_object_id: str | None = None
    target_object_cluster_id: str | None = None
    priority: Priority = "medium"
    reason: str

    @field_validator("cluster_id", "object_id", "region_id", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator(
        "semantic_anchor_region_id", "target_object_id", "target_object_cluster_id"
    )
    @classmethod
    def _optional_non_empty_str(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @model_validator(mode="after")
    def _validate_target_logic(self) -> "VisualObjectRegionDirective":
        targets_object = self.facing_intent in {
            "face_target",
            "face_away_from_target",
        } or self.placement_role in {"paired_seat", "functional_support"}
        if targets_object and self.target_object_id is None:
            raise ValueError(
                "target_object_id is required for target-facing or paired objects"
            )
        if (
            self.target_object_id is not None
            and self.target_object_id == self.object_id
            and self.target_object_cluster_id in {None, self.cluster_id}
        ):
            raise ValueError("target object must differ from source object")
        return self


class VisualObjectViewingPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    viewer_cluster_id: str
    viewer_object_id: str
    target_cluster_id: str
    target_object_id: str
    pair_type: ViewingPairType = "viewer_faces_target"
    priority: Priority = "high"
    reason: str

    @field_validator(
        "viewer_cluster_id",
        "viewer_object_id",
        "target_cluster_id",
        "target_object_id",
        "reason",
    )
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @model_validator(mode="after")
    def _source_target_differ(self) -> "VisualObjectViewingPair":
        if (
            self.viewer_cluster_id == self.target_cluster_id
            and self.viewer_object_id == self.target_object_id
        ):
            raise ValueError("viewer and target objects must differ")
        return self


class VisualClusterRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    relation: RelationType
    priority: Priority = "medium"
    reason: str

    @field_validator("a", "b", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @model_validator(mode="after")
    def _a_b_not_same(self) -> "VisualClusterRelation":
        if self.a == self.b:
            raise ValueError("cluster relation endpoints must differ")
        return self


class VisualRelationPlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status
    concept_id: str = "visual_concept_01"
    concept_family: ConceptFamily = "zoned"
    primary_cluster_id: str
    secondary_cluster_id: str | None = None
    cluster_zone_plan: list[VisualClusterZoneDirective] = Field(default_factory=list)
    object_orientations: list[VisualObjectOrientationDirective] = Field(
        default_factory=list
    )
    object_region_plan: list[VisualObjectRegionDirective] = Field(default_factory=list)
    object_viewing_pairs: list[VisualObjectViewingPair] = Field(default_factory=list)
    cluster_relations: list[VisualClusterRelation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @field_validator("concept_id", "primary_cluster_id")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("secondary_cluster_id")
    @classmethod
    def _optional_non_empty_str(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @field_validator("notes", "missing")
    @classmethod
    def _strip_string_lists(cls, value: list[str], info) -> list[str]:
        return [_strip_required_str(item, info.field_name) for item in value]

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> "VisualRelationPlannerOutput":
        cluster_ids = [item.cluster_id for item in self.cluster_zone_plan]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("cluster_zone_plan must be unique by cluster_id")

        object_keys = [
            (item.cluster_id, item.object_id) for item in self.object_orientations
        ]
        if len(object_keys) != len(set(object_keys)):
            raise ValueError(
                "object_orientations must be unique by (cluster_id, object_id)"
            )

        object_region_keys = [
            (item.cluster_id, item.object_id) for item in self.object_region_plan
        ]
        if len(object_region_keys) != len(set(object_region_keys)):
            raise ValueError(
                "object_region_plan must be unique by (cluster_id, object_id)"
            )

        relation_keys = [
            tuple(sorted((item.a, item.b))) for item in self.cluster_relations
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("cluster_relations must not contain duplicate pairs")

        if (
            self.secondary_cluster_id is not None
            and self.secondary_cluster_id == self.primary_cluster_id
        ):
            raise ValueError("secondary_cluster_id must differ from primary_cluster_id")
        if self.status != "NEED_INFO" and self.missing:
            raise ValueError("missing must be empty unless status=NEED_INFO")
        return self
