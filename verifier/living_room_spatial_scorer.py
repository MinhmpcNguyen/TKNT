from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agent_schema.stylist_schema import BBox, Point, StyledObject, StylistOutput


@dataclass(frozen=True)
class ScoringConfig:
    object_against_wall_mm: int = 200
    sofa_center_max_ratio_of_room_diag: float = 0.45
    sofa_blocks_path_buffer_mm: int = 0
    coffee_table_gap_min_mm: int = 200
    coffee_table_gap_max_mm: int = 1200
    coffee_table_centerline_tol_mm: int = 700
    tv_console_min_distance_to_sofa_mm: int = 1200
    tv_console_axis_overlap_ratio_min: float = 0.3
    tv_console_near_window_mm: int = 800
    tv_with_console_max_center_distance_mm: int = 1200
    tv_min_distance_to_window_mm: int = 1000
    ceiling_lamp_near_living_center_ratio_of_room_diag: float = 0.4
    ceiling_lamp_near_object_max_mm: int = 500
    sofa_window_clearance_min_mm: int = 0
    rug_min_intersection_ratio: float = 0.05
    seat_conversation_max_center_distance_mm: int = 2600
    armchair_near_coffee_table_mm: int = 2000
    side_table_near_seat_mm: int = 900


RuleStatus = Literal["passed", "failed", "skipped"]
ScoreStatus = Literal["OK", "NOT_APPLICABLE", "NEED_INFO"]
Relation = Literal["less", "equal", "greater"]


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    name: str
    weight: float
    earned: float
    status: RuleStatus
    details: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "weight": self.weight,
            "earned": self.earned,
            "status": self.status,
            "details": self.details,
        }


def _force_no_skipped_rules(rules: list[RuleResult]) -> list[RuleResult]:
    out: list[RuleResult] = []
    for r in rules:
        if r.status != "skipped":
            out.append(r)
            continue
        out.append(
            RuleResult(
                rule_id=r.rule_id,
                name=r.name,
                weight=r.weight,
                earned=0,
                status="failed",
                details=r.details,
            )
        )
    return out


@dataclass(frozen=True)
class OobResult:
    status: RuleStatus
    scene_invalid: bool | None
    oob_rate: float | None
    checked_instance_ids: list[str] = field(default_factory=list)
    out_of_bounds_instance_ids: list[str] = field(default_factory=list)
    out_of_bounds_objects: list[dict[str, str]] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "scene_invalid": self.scene_invalid,
            "oob_rate": self.oob_rate,
            "checked_instance_ids": self.checked_instance_ids,
            "out_of_bounds_instance_ids": self.out_of_bounds_instance_ids,
            "out_of_bounds_objects": self.out_of_bounds_objects,
            "details": self.details,
        }


@dataclass(frozen=True)
class BblResult:
    status: RuleStatus
    mean_intersection_area_mm2: float | None
    pair_count: int
    checked_instance_ids: list[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mean_intersection_area_mm2": self.mean_intersection_area_mm2,
            "pair_count": self.pair_count,
            "checked_instance_ids": self.checked_instance_ids,
            "details": self.details,
        }


@dataclass(frozen=True)
class ComparisonResult:
    left: str
    right: str
    predicted_relation: Relation
    allowed_relations: list[Relation]
    correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "predicted_relation": self.predicted_relation,
            "allowed_relations": self.allowed_relations,
            "correct": self.correct,
        }


@dataclass(frozen=True)
class NumericalResult:
    status: RuleStatus
    gt_spec: dict[str, dict[str, int]] = field(default_factory=dict)
    pred_counts: dict[str, int] = field(default_factory=dict)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    accuracy: float | None = None
    comparison_accuracy: float | None = None
    comparisons: list[ComparisonResult] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "gt_spec": self.gt_spec,
            "pred_counts": self.pred_counts,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "comparison_accuracy": self.comparison_accuracy,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "details": self.details,
        }


@dataclass(frozen=True)
class ScoreResult:
    status: ScoreStatus
    room_type: str | None
    overall_score: float
    max_score: float
    normalized_score_0_100: float | None
    rules: list[RuleResult] = field(default_factory=list)
    object_instances: dict[str, list[str]] = field(default_factory=dict)
    oob: OobResult | None = None
    bbl: BblResult | None = None
    numerical: NumericalResult | None = None

    def to_dict(self) -> dict[str, object]:
        numerical_payload = (
            self.numerical.to_dict() if self.numerical is not None else None
        )
        return {
            "status": self.status,
            "room_type": self.room_type,
            "overall_score": self.overall_score,
            "max_score": self.max_score,
            "normalized_score_0_100": self.normalized_score_0_100,
            "rules": [r.to_dict() for r in self.rules],
            "object_instances": self.object_instances,
            "oob": (self.oob.to_dict() if self.oob is not None else None),
            "bbl": (self.bbl.to_dict() if self.bbl is not None else None),
            "numerical": numerical_payload,
            "numerical_reasoning": numerical_payload,
        }


_CANONICAL_OBJECT_TYPES: dict[str, set[str]] = {
    "sofa": {"sofa", "couch", "sectional"},
    "coffee_table": {"coffee_table"},
    "tv_console": {"tv_console", "media_console", "tv_stand"},
    "tv": {"tv", "television"},
    "ceiling_lamp": {
        "ceiling_lamp",
        "ceiling_light",
        "celling_lamp",
        "celling_light",
        "chandelier",
        "pendant_light",
    },
    "curtain": {"curtain", "blinds", "curtain_or_blinds"},
    "rug": {"rug", "carpet"},
    "armchair": {"armchair"},
    "side_table": {"side_table", "end_table"},
}

_LIVING_ROOM_NUMERICAL_SPEC: dict[str, tuple[int, int]] = {
    "sofa": (1, 1),
    "coffee_table": (1, 1),
    "tv_console": (1, 1),
    "tv": (1, 1),
    "ceiling_lamp": (1, 1),
    "curtain": (1, 1),
    "rug": (1, 1),
    "armchair": (1, 1),
    "side_table": (1, 1),
}

_LIVING_ROOM_COMPARISONS: list[tuple[str, str]] = [
    ("tv", "tv_console"),
    ("coffee_table", "sofa"),
]


def score_living_room_spatial(
    stylist_output: StylistOutput,
    *,
    config: ScoringConfig | None = None,
) -> ScoreResult:
    cfg = config or ScoringConfig()

    objects_by_cat = _objects_by_category(stylist_output.objects)
    living_key_objects = _living_room_key_objects(objects_by_cat)
    room_poly = stylist_output.room.polygon_ccw if stylist_output.room else []

    oob = (
        _compute_oob(room_poly, stylist_output.objects)
        if stylist_output.room
        else _oob_skipped("Room missing.")
    )
    bbl = _compute_bbl(living_key_objects)
    numerical = _compute_numerical(
        objects_by_cat, _LIVING_ROOM_NUMERICAL_SPEC, _LIVING_ROOM_COMPARISONS
    )
    object_instances = _object_instances_by_category(stylist_output.objects)

    if stylist_output.status != "OK" or stylist_output.room is None:
        return ScoreResult(
            status="NEED_INFO",
            room_type=stylist_output.room.room_type if stylist_output.room else None,
            overall_score=0.0,
            max_score=0.0,
            normalized_score_0_100=None,
            rules=[],
            object_instances=object_instances,
            oob=oob,
            bbl=bbl,
            numerical=numerical,
        )

    room_type = (stylist_output.room.room_type or "").strip().lower()
    if room_type != "living_room":
        return ScoreResult(
            status="NOT_APPLICABLE",
            room_type=stylist_output.room.room_type,
            overall_score=0.0,
            max_score=0.0,
            normalized_score_0_100=None,
            rules=[],
            object_instances=object_instances,
            oob=_oob_skipped("Not applicable (room_type is not living_room)."),
            bbl=bbl,
            numerical=numerical,
        )

    room_centroid = _polygon_centroid(room_poly) if len(room_poly) >= 3 else (0.0, 0.0)
    room_diag = _room_diag(room_poly)

    sofa = _primary(objects_by_cat.get("sofa", []))
    coffee_table = _primary(objects_by_cat.get("coffee_table", []))
    tv_console = _primary(objects_by_cat.get("tv_console", []))
    tv = _primary(objects_by_cat.get("tv", []))
    ceiling_lamp = _primary(objects_by_cat.get("ceiling_lamp", []))
    curtains = objects_by_cat.get("curtain", [])
    rug = _primary(objects_by_cat.get("rug", []))
    armchair = _primary(objects_by_cat.get("armchair", []))
    side_table = _primary(objects_by_cat.get("side_table", []))

    door_segments = _opening_segments(
        (stylist_output.room.openings.doors if stylist_output.room.openings else [])
    )
    window_segments = _opening_segments(
        (stylist_output.room.openings.windows if stylist_output.room.openings else [])
    )
    obstacle_bboxes = _obstacle_bboxes(getattr(stylist_output.room, "obstacles", []))

    rules: list[RuleResult] = []

    # Sofa
    rules.append(_rule_sofa_central(sofa, room_centroid, room_diag, cfg))
    rules.append(_rule_sofa_faces_media(sofa, tv, tv_console, room_poly))
    rules.append(
        _rule_against_wall(
            "sofa/against_wall", "Sofa near a wall", 7, sofa, room_poly, cfg
        )
    )
    rules.append(_rule_sofa_not_block_path(sofa, door_segments, room_centroid, cfg))

    # Coffee table
    rules.append(
        _rule_coffee_table_before_sofa(
            sofa, coffee_table, tv_console, room_centroid, cfg
        )
    )
    rules.append(
        _rule_coffee_table_in_seating_cluster(sofa, armchair, coffee_table, cfg)
    )
    rules.append(_rule_coffee_table_distance_to_sofa(sofa, coffee_table, cfg))

    # TV console
    rules.append(
        _rule_media_opposite_sofa(
            "tv_console/opposite_sofa",
            "TV console opposite sofa",
            9,
            sofa,
            tv_console,
            cfg,
        )
    )
    rules.append(
        _rule_against_wall(
            "tv_console/against_wall",
            "TV console near a wall",
            8,
            tv_console,
            room_poly,
            cfg,
        )
    )
    rules.append(
        _rule_tv_console_not_block_window_or_path(
            tv_console,
            door_segments,
            window_segments,
            room_centroid,
            obstacle_bboxes,
            cfg,
        )
    )

    # TV
    rules.append(_rule_tv_with_console(tv, tv_console, cfg))
    rules.append(
        _rule_media_opposite_sofa(
            "tv/opposite_sofa", "TV opposite sofa", 9, sofa, tv, cfg
        )
    )
    rules.append(_rule_tv_avoid_window_glare(tv, window_segments, cfg))

    # Ceiling lamp
    rules.append(_rule_ceiling_lamp_general_lighting(ceiling_lamp))
    rules.append(
        _rule_ceiling_lamp_near_living_center(
            ceiling_lamp, sofa, coffee_table, room_centroid, room_diag, cfg
        )
    )
    rules.append(
        _rule_ceiling_lamp_not_tied_to_object(ceiling_lamp, [sofa, tv, tv_console], cfg)
    )

    # Curtain
    rules.append(_rule_curtain_present_with_window(curtains, window_segments))
    rules.append(
        _rule_curtain_constraints_on_media(
            sofa, tv, tv_console, window_segments, obstacle_bboxes, cfg
        )
    )
    rules.append(
        _rule_sofa_not_block_curtain_zone(sofa, window_segments, obstacle_bboxes, cfg)
    )

    # Rug
    rules.append(_rule_rug_under_seating_cluster(rug, sofa, coffee_table, cfg))
    rules.append(_rule_rug_not_isolated(rug, sofa, coffee_table))
    rules.append(_rule_coffee_table_on_rug(rug, coffee_table))

    # Armchair
    rules.append(_rule_armchair_in_cluster(armchair, sofa, coffee_table, cfg))
    rules.append(
        _rule_armchair_typical_position(armchair, sofa, coffee_table, tv_console, cfg)
    )
    rules.append(
        _rule_object_not_block_path_optional(
            "armchair/not_block_path",
            "Armchair does not block main path",
            10,
            armchair,
            door_segments,
            room_centroid,
        )
    )

    # Side table
    rules.append(_rule_side_table_next_to_seat(side_table, sofa, armchair, cfg))
    rules.append(
        _rule_side_table_not_replacing_coffee_table(
            side_table, sofa, armchair, coffee_table, cfg
        )
    )

    rules = _force_no_skipped_rules(rules)
    earned_total = sum(r.earned for r in rules)
    max_total = sum(r.weight for r in rules)
    normalized = (earned_total / max_total * 100.0) if max_total > 0 else None

    return ScoreResult(
        status="OK",
        room_type=stylist_output.room.room_type,
        overall_score=round(earned_total, 3),
        max_score=round(max_total, 3),
        normalized_score_0_100=(
            round(normalized, 3) if normalized is not None else None
        ),
        rules=rules,
        object_instances=object_instances,
        oob=oob,
        bbl=bbl,
        numerical=numerical,
    )


def load_stylist_output_json(path: str | Path) -> StylistOutput:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stylist_payload = _extract_stylist_payload(payload)
    return StylistOutput.model_validate(stylist_payload)


def _extract_stylist_payload(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        if isinstance(payload.get("final_output"), dict):
            return payload["final_output"]
        if isinstance(payload.get("status"), str) and isinstance(
            payload.get("objects"), list
        ):
            return payload
    raise ValueError(
        "Input JSON is not a stylist output (expected keys: status, room, objects)"
    )


def _objects_by_category(objects: list[StyledObject]) -> dict[str, list[StyledObject]]:
    grouped: dict[str, list[StyledObject]] = {k: [] for k in _CANONICAL_OBJECT_TYPES}
    for obj in objects:
        cat = _canonical_category(obj.object_type)
        if cat is not None:
            grouped[cat].append(obj)
    return grouped


def _object_instances_by_category(objects: list[StyledObject]) -> dict[str, list[str]]:
    grouped = _objects_by_category(objects)
    return {cat: [o.instance_id for o in objs] for cat, objs in grouped.items() if objs}


def _canonical_category(object_type: str) -> str | None:
    ot = (object_type or "").strip().lower()
    for cat, allowed in _CANONICAL_OBJECT_TYPES.items():
        if ot in allowed:
            return cat
    return None


def _primary(objects: list[StyledObject]) -> StyledObject | None:
    if not objects:
        return None

    def area(o: StyledObject) -> int:
        w = max(0, o.bbox.max_x - o.bbox.min_x)
        h = max(0, o.bbox.max_y - o.bbox.min_y)
        return w * h

    return max(objects, key=area)


def _rule_sofa_central(
    sofa: StyledObject | None,
    room_centroid: tuple[float, float],
    room_diag: float,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 10
    if sofa is None:
        return RuleResult(
            "sofa/is_seating_center",
            "Sofa is seating center",
            weight,
            0,
            "failed",
            "No sofa found.",
        )
    diag = room_diag if room_diag > 0 else 1.0
    threshold = cfg.sofa_center_max_ratio_of_room_diag * diag
    d = _euclidean(_bbox_center(sofa.bbox), room_centroid)
    passed = d <= threshold
    return RuleResult(
        "sofa/is_seating_center",
        "Sofa is seating center",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"dist_to_center_mm={round(d, 1)} (threshold={round(threshold, 1)}).",
    )


def _rule_against_wall(
    rule_id: str,
    name: str,
    weight: float,
    obj: StyledObject | None,
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    if obj is None:
        return RuleResult(rule_id, name, weight, 0, "failed", "Object missing.")
    if len(room_poly) < 3:
        return RuleResult(
            rule_id, name, weight, 0, "skipped", "Room polygon missing/invalid."
        )
    d = _min_distance_bbox_to_polygon(obj.bbox, room_poly)
    passed = d <= cfg.object_against_wall_mm
    return RuleResult(
        rule_id,
        name,
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"min_distance_to_wall_mm={round(d, 1)} (threshold={cfg.object_against_wall_mm}).",
    )


def _rule_sofa_faces_media(
    sofa: StyledObject | None,
    tv: StyledObject | None,
    tv_console: StyledObject | None,
    room_poly: list[Point],
) -> RuleResult:
    weight = 9
    if sofa is None:
        return RuleResult(
            "sofa/faces_tv_or_main_view",
            "Sofa faces TV or main view",
            weight,
            0,
            "failed",
            "No sofa found.",
        )
    media = tv or tv_console
    if media is None:
        return RuleResult(
            "sofa/faces_tv_or_main_view",
            "Sofa faces TV or main view",
            weight,
            0,
            "skipped",
            "Missing tv/tv_console.",
        )
    if len(room_poly) < 3:
        return RuleResult(
            "sofa/faces_tv_or_main_view",
            "Sofa faces TV or main view",
            weight,
            0,
            "skipped",
            "Room polygon missing/invalid.",
        )

    front = _infer_object_front_direction_from_wall(sofa.bbox, room_poly)
    if front is None:
        return RuleResult(
            "sofa/faces_tv_or_main_view",
            "Sofa faces TV or main view",
            weight,
            0,
            "failed",
            "Cannot infer sofa front direction.",
        )
    sofa_center = _bbox_center(sofa.bbox)
    media_center = _bbox_center(media.bbox)
    vec = (media_center[0] - sofa_center[0], media_center[1] - sofa_center[1])
    dot = vec[0] * front[0] + vec[1] * front[1]
    passed = dot > 0
    return RuleResult(
        "sofa/faces_tv_or_main_view",
        "Sofa faces TV or main view",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"front_dir={front}; dot_to_media={round(dot, 1)}; media={media.instance_id}.",
    )


def _rule_sofa_not_block_path(
    sofa: StyledObject | None,
    door_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    room_centroid: tuple[float, float],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 10
    if sofa is None:
        return RuleResult(
            "sofa/not_blocking_main_path",
            "Sofa does not block main path",
            weight,
            0,
            "failed",
            "No sofa found.",
        )
    if not door_segments:
        return RuleResult(
            "sofa/not_blocking_main_path",
            "Sofa does not block main path",
            weight,
            0,
            "skipped",
            "No door opening segment found.",
        )

    door_mid = _segment_midpoint(door_segments[0][1], door_segments[0][2])
    blocked_center = _segment_intersects_rect(
        door_mid,
        room_centroid,
        _inflate_bbox(sofa.bbox, cfg.sofa_blocks_path_buffer_mm),
    )
    passed = not blocked_center
    return RuleResult(
        "sofa/not_blocking_main_path",
        "Sofa does not block main path",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"door_to_centroid_intersects={blocked_center}.",
    )


def _rule_coffee_table_before_sofa(
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
    tv_console: StyledObject | None,
    room_centroid: tuple[float, float],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 10
    if coffee_table is None:
        return RuleResult(
            "coffee_table/in_front_of_sofa",
            "Coffee table in front of sofa",
            weight,
            0,
            "failed",
            "No coffee_table found.",
        )
    if sofa is None:
        return RuleResult(
            "coffee_table/in_front_of_sofa",
            "Coffee table in front of sofa",
            weight,
            0,
            "skipped",
            "No sofa found.",
        )

    sofa_center = _bbox_center(sofa.bbox)
    table_center = _bbox_center(coffee_table.bbox)
    media_center = (
        _bbox_center(tv_console.bbox) if tv_console is not None else room_centroid
    )
    d_to_axis = _distance_point_to_segment(table_center, sofa_center, media_center)
    passed = d_to_axis <= cfg.coffee_table_centerline_tol_mm
    return RuleResult(
        "coffee_table/in_front_of_sofa",
        "Coffee table in front of sofa",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"dist_to_sofa_media_axis_mm={round(d_to_axis, 1)} (tol={cfg.coffee_table_centerline_tol_mm}).",
    )


def _rule_coffee_table_in_seating_cluster(
    sofa: StyledObject | None,
    armchair: StyledObject | None,
    coffee_table: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if coffee_table is None:
        return RuleResult(
            "coffee_table/centered_in_seating_cluster",
            "Coffee table centered in seating cluster",
            weight,
            0,
            "failed",
            "No coffee_table found.",
        )
    if sofa is None:
        return RuleResult(
            "coffee_table/centered_in_seating_cluster",
            "Coffee table centered in seating cluster",
            weight,
            0,
            "skipped",
            "No sofa found.",
        )

    seats = [sofa]
    if armchair is not None:
        seats.append(armchair)
    seat_bbox = _bbox_union([s.bbox for s in seats])
    expanded = _inflate_bbox(seat_bbox, 300)
    table_center = _bbox_center(coffee_table.bbox)
    passed = _point_in_bbox(table_center, expanded)
    return RuleResult(
        "coffee_table/centered_in_seating_cluster",
        "Coffee table centered in seating cluster",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"table_center_in_seat_union={passed}.",
    )


def _rule_coffee_table_distance_to_sofa(
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if coffee_table is None:
        return RuleResult(
            "coffee_table/not_too_far_from_sofa",
            "Coffee table not too far from sofa",
            weight,
            0,
            "failed",
            "No coffee_table found.",
        )
    if sofa is None:
        return RuleResult(
            "coffee_table/not_too_far_from_sofa",
            "Coffee table not too far from sofa",
            weight,
            0,
            "skipped",
            "No sofa found.",
        )

    gap = _min_gap_between_bboxes(sofa.bbox, coffee_table.bbox)
    passed = cfg.coffee_table_gap_min_mm <= gap <= cfg.coffee_table_gap_max_mm
    return RuleResult(
        "coffee_table/not_too_far_from_sofa",
        "Coffee table not too far from sofa",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"gap_sofa_to_table_mm={round(gap, 1)} (min={cfg.coffee_table_gap_min_mm}, max={cfg.coffee_table_gap_max_mm}).",
    )


def _rule_media_opposite_sofa(
    rule_id: str,
    name: str,
    weight: float,
    sofa: StyledObject | None,
    media: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    if media is None:
        return RuleResult(rule_id, name, weight, 0, "failed", "Media object missing.")
    if sofa is None:
        return RuleResult(rule_id, name, weight, 0, "skipped", "No sofa found.")

    overlap_ratio = _max_axis_overlap_ratio(sofa.bbox, media.bbox)
    dist = _euclidean(_bbox_center(sofa.bbox), _bbox_center(media.bbox))
    passed = (
        overlap_ratio >= cfg.tv_console_axis_overlap_ratio_min
        and dist >= cfg.tv_console_min_distance_to_sofa_mm
    )
    return RuleResult(
        rule_id,
        name,
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"max_axis_overlap_ratio={round(overlap_ratio, 3)} (min={cfg.tv_console_axis_overlap_ratio_min}); dist_mm={round(dist, 1)}.",
    )


def _rule_tv_console_not_block_window_or_path(
    tv_console: StyledObject | None,
    door_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    room_centroid: tuple[float, float],
    obstacle_bboxes: dict[str, list[BBox]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if tv_console is None:
        return RuleResult(
            "tv_console/not_blocking_window_or_path",
            "TV console does not block window or main path",
            weight,
            0,
            "failed",
            "No tv_console found.",
        )

    blockers: list[str] = []
    if door_segments:
        door_mid = _segment_midpoint(door_segments[0][1], door_segments[0][2])
        if _segment_intersects_rect(door_mid, room_centroid, tv_console.bbox):
            blockers.append("door_to_centroid")

    for ob in obstacle_bboxes.get("window_clearance", []):
        if _rects_intersect(tv_console.bbox, ob):
            blockers.append("window_clearance_overlap")
            break

    if window_segments:
        d = min(
            _distance_point_to_segment(_bbox_center(tv_console.bbox), a, b)
            for _, a, b in window_segments
        )
        if d <= cfg.tv_console_near_window_mm:
            blockers.append(f"near_window_mm={round(d, 1)}")

    passed = not blockers
    return RuleResult(
        "tv_console/not_blocking_window_or_path",
        "TV console does not block window or main path",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"issues={blockers}.",
    )


def _rule_tv_with_console(
    tv: StyledObject | None, tv_console: StyledObject | None, cfg: ScoringConfig
) -> RuleResult:
    weight = 10
    if tv is None:
        return RuleResult(
            "tv/with_tv_console",
            "TV is placed with TV console / media wall",
            weight,
            0,
            "failed",
            "No tv found.",
        )
    if tv_console is None:
        return RuleResult(
            "tv/with_tv_console",
            "TV is placed with TV console / media wall",
            weight,
            0,
            "skipped",
            "No tv_console found.",
        )

    if (
        tv.place_on is not None
        and tv.place_on.target_instance_id == tv_console.instance_id
    ):
        return RuleResult(
            "tv/with_tv_console",
            "TV is placed with TV console / media wall",
            weight,
            weight,
            "passed",
            "TV place_on points to tv_console.",
        )

    d = _euclidean(_bbox_center(tv.bbox), _bbox_center(tv_console.bbox))
    passed = d <= cfg.tv_with_console_max_center_distance_mm
    return RuleResult(
        "tv/with_tv_console",
        "TV is placed with TV console / media wall",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"center_distance_tv_to_console_mm={round(d, 1)} (max={cfg.tv_with_console_max_center_distance_mm}).",
    )


def _rule_tv_avoid_window_glare(
    tv: StyledObject | None,
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if tv is None:
        return RuleResult(
            "tv/avoid_direct_window_glare",
            "TV avoids direct window glare",
            weight,
            0,
            "failed",
            "No tv found.",
        )
    if not window_segments:
        return RuleResult(
            "tv/avoid_direct_window_glare",
            "TV avoids direct window glare",
            weight,
            0,
            "skipped",
            "No window opening segment found.",
        )

    tv_center = _bbox_center(tv.bbox)
    d = min(_distance_point_to_segment(tv_center, a, b) for _, a, b in window_segments)
    passed = d >= cfg.tv_min_distance_to_window_mm
    return RuleResult(
        "tv/avoid_direct_window_glare",
        "TV avoids direct window glare",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"min_tv_to_window_mm={round(d, 1)} (min={cfg.tv_min_distance_to_window_mm}).",
    )


def _rule_ceiling_lamp_general_lighting(
    ceiling_lamp: StyledObject | None,
) -> RuleResult:
    weight = 10
    if ceiling_lamp is None:
        return RuleResult(
            "ceiling_lamp/general_lighting",
            "Ceiling lamp provides general lighting",
            weight,
            0,
            "failed",
            "No ceiling_lamp found.",
        )
    placed = (
        ceiling_lamp.place_on is not None
        and ceiling_lamp.place_on.method == "hang_on"
        and ceiling_lamp.place_on.target_instance_id == "__ceiling__"
    )
    return RuleResult(
        "ceiling_lamp/general_lighting",
        "Ceiling lamp provides general lighting",
        weight,
        (weight if placed else 0),
        ("passed" if placed else "failed"),
        f"place_on={ceiling_lamp.place_on.model_dump() if ceiling_lamp.place_on else None}.",
    )


def _rule_ceiling_lamp_near_living_center(
    ceiling_lamp: StyledObject | None,
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
    room_centroid: tuple[float, float],
    room_diag: float,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if ceiling_lamp is None:
        return RuleResult(
            "ceiling_lamp/near_activity_center",
            "Ceiling lamp near living area center",
            weight,
            0,
            "failed",
            "No ceiling_lamp found.",
        )

    if sofa is not None and coffee_table is not None:
        living_center = _avg_point(
            [_bbox_center(sofa.bbox), _bbox_center(coffee_table.bbox)]
        )
    elif sofa is not None:
        living_center = _bbox_center(sofa.bbox)
    else:
        living_center = room_centroid

    diag = room_diag if room_diag > 0 else 1.0
    threshold = cfg.ceiling_lamp_near_living_center_ratio_of_room_diag * diag
    d = _euclidean(_bbox_center(ceiling_lamp.bbox), living_center)
    passed = d <= threshold
    return RuleResult(
        "ceiling_lamp/near_activity_center",
        "Ceiling lamp near living area center",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"dist_mm={round(d, 1)}; threshold_mm={round(threshold, 1)}; diag_mm={round(diag, 1)}.",
    )


def _rule_ceiling_lamp_not_tied_to_object(
    ceiling_lamp: StyledObject | None,
    anchors: list[StyledObject | None],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 7
    if ceiling_lamp is None:
        return RuleResult(
            "ceiling_lamp/not_paired_to_sofa_or_tv",
            "Ceiling lamp not tied to a single object",
            weight,
            0,
            "failed",
            "No ceiling_lamp found.",
        )

    anchor_boxes = [a.bbox for a in anchors if a is not None]
    if not anchor_boxes:
        return RuleResult(
            "ceiling_lamp/not_paired_to_sofa_or_tv",
            "Ceiling lamp not tied to a single object",
            weight,
            0,
            "skipped",
            "No anchor objects found.",
        )

    center = _bbox_center(ceiling_lamp.bbox)
    over_anchor = any(
        _point_in_bbox_padded(
            center, bbox, padding_mm=cfg.ceiling_lamp_near_object_max_mm
        )
        for bbox in anchor_boxes
    )
    passed = not over_anchor
    return RuleResult(
        "ceiling_lamp/not_paired_to_sofa_or_tv",
        "Ceiling lamp not tied to a single object",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"over_anchor={over_anchor}; padding_mm={cfg.ceiling_lamp_near_object_max_mm}.",
    )


def _rule_curtain_present_with_window(
    curtains: list[StyledObject],
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> RuleResult:
    weight = 10
    if not window_segments:
        return RuleResult(
            "curtain/with_window",
            "Curtain/blinds present when window exists",
            weight,
            0,
            "skipped",
            "No window opening segment found.",
        )
    if not curtains:
        return RuleResult(
            "curtain/with_window",
            "Curtain/blinds present when window exists",
            weight,
            0,
            "failed",
            "No curtain/blinds found.",
        )

    window_ids = {wid for wid, _, _ in window_segments}
    attached = [
        c
        for c in curtains
        if c.place_on is not None
        and c.place_on.method == "hang_on"
        and c.place_on.target_instance_id in window_ids
    ]
    passed = bool(attached)
    return RuleResult(
        "curtain/with_window",
        "Curtain/blinds present when window exists",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"curtains={len(curtains)}; attached_to_window={len(attached)}.",
    )


def _rule_curtain_constraints_on_media(
    sofa: StyledObject | None,
    tv: StyledObject | None,
    tv_console: StyledObject | None,
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    obstacle_bboxes: dict[str, list[BBox]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if not window_segments:
        return RuleResult(
            "curtain/influences_sofa_and_tv_placement",
            "Curtains constrain sofa/TV placement",
            weight,
            0,
            "skipped",
            "No window opening segment found.",
        )
    if tv is None and tv_console is None and sofa is None:
        return RuleResult(
            "curtain/influences_sofa_and_tv_placement",
            "Curtains constrain sofa/TV placement",
            weight,
            0,
            "skipped",
            "No sofa/tv/tv_console found.",
        )

    offenders: list[str] = []
    for ob in obstacle_bboxes.get("window_clearance", []):
        if sofa is not None and _rects_intersect(sofa.bbox, ob):
            offenders.append(f"sofa_overlaps_window_clearance:{sofa.instance_id}")
        if tv_console is not None and _rects_intersect(tv_console.bbox, ob):
            offenders.append(
                f"tv_console_overlaps_window_clearance:{tv_console.instance_id}"
            )
        if tv is not None and _rects_intersect(tv.bbox, ob):
            offenders.append(f"tv_overlaps_window_clearance:{tv.instance_id}")

    if tv is not None:
        tv_center = _bbox_center(tv.bbox)
        d = min(
            _distance_point_to_segment(tv_center, a, b) for _, a, b in window_segments
        )
        if d < cfg.tv_min_distance_to_window_mm:
            offenders.append(f"tv_near_window_mm={round(d, 1)}")

    passed = not offenders
    return RuleResult(
        "curtain/influences_sofa_and_tv_placement",
        "Curtains constrain sofa/TV placement",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"issues={offenders}.",
    )


def _rule_sofa_not_block_curtain_zone(
    sofa: StyledObject | None,
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    obstacle_bboxes: dict[str, list[BBox]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 7
    if not window_segments:
        return RuleResult(
            "sofa/window_clearance",
            "Sofa does not block curtain zone",
            weight,
            0,
            "skipped",
            "No window opening segment found.",
        )
    if sofa is None:
        return RuleResult(
            "sofa/window_clearance",
            "Sofa does not block curtain zone",
            weight,
            0,
            "skipped",
            "No sofa found.",
        )

    overlaps = any(
        _rects_intersect(sofa.bbox, ob)
        for ob in obstacle_bboxes.get("window_clearance", [])
    )
    if overlaps:
        return RuleResult(
            "sofa/window_clearance",
            "Sofa does not block curtain zone",
            weight,
            0,
            "failed",
            "Sofa overlaps window_clearance obstacle.",
        )

    sofa_center = _bbox_center(sofa.bbox)
    d = min(
        _distance_point_to_segment(sofa_center, a, b) for _, a, b in window_segments
    )
    return RuleResult(
        "sofa/window_clearance",
        "Sofa does not block curtain zone",
        weight,
        weight,
        "passed",
        f"min_sofa_to_window_mm={round(d, 1)}; clearance_ok=True.",
    )


def _rule_rug_under_seating_cluster(
    rug: StyledObject | None,
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if rug is None:
        return RuleResult(
            "rug/under_sofa_and_coffee_table_cluster",
            "Rug under sofa + coffee table",
            weight,
            0,
            "failed",
            "No rug found.",
        )
    if sofa is None or coffee_table is None:
        return RuleResult(
            "rug/under_sofa_and_coffee_table_cluster",
            "Rug under sofa + coffee table",
            weight,
            0,
            "skipped",
            "Missing sofa or coffee_table.",
        )

    sofa_overlap = _intersection_area_ratio(rug.bbox, sofa.bbox)
    table_inside = _point_in_bbox(_bbox_center(coffee_table.bbox), rug.bbox)
    passed = table_inside and sofa_overlap >= cfg.rug_min_intersection_ratio
    return RuleResult(
        "rug/under_sofa_and_coffee_table_cluster",
        "Rug under sofa + coffee table",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"coffee_table_center_in_rug={table_inside}; rug_sofa_intersection_ratio={round(sofa_overlap, 3)}.",
    )


def _rule_rug_not_isolated(
    rug: StyledObject | None,
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
) -> RuleResult:
    weight = 8
    if rug is None:
        return RuleResult(
            "rug/part_of_seating_zone",
            "Rug belongs to seating area",
            weight,
            0,
            "failed",
            "No rug found.",
        )
    if sofa is None and coffee_table is None:
        return RuleResult(
            "rug/part_of_seating_zone",
            "Rug belongs to seating area",
            weight,
            0,
            "skipped",
            "No sofa or coffee_table.",
        )

    hits = False
    if sofa is not None and _rects_intersect(rug.bbox, sofa.bbox):
        hits = True
    if coffee_table is not None and _rects_intersect(rug.bbox, coffee_table.bbox):
        hits = True
    return RuleResult(
        "rug/part_of_seating_zone",
        "Rug belongs to seating area",
        weight,
        (weight if hits else 0),
        ("passed" if hits else "failed"),
        f"intersects_sofa_or_coffee_table={hits}.",
    )


def _rule_coffee_table_on_rug(
    rug: StyledObject | None, coffee_table: StyledObject | None
) -> RuleResult:
    weight = 8
    if coffee_table is None:
        return RuleResult(
            "coffee_table/on_rug",
            "Coffee table stands on rug",
            weight,
            0,
            "failed",
            "No coffee_table found.",
        )
    if rug is None:
        return RuleResult(
            "coffee_table/on_rug",
            "Coffee table stands on rug",
            weight,
            0,
            "failed",
            "No rug found.",
        )

    inside = _point_in_bbox(_bbox_center(coffee_table.bbox), rug.bbox)
    return RuleResult(
        "coffee_table/on_rug",
        "Coffee table stands on rug",
        weight,
        (weight if inside else 0),
        ("passed" if inside else "failed"),
        f"coffee_table_center_in_rug={inside}.",
    )


def _rule_armchair_in_cluster(
    armchair: StyledObject | None,
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if armchair is None:
        return RuleResult(
            "armchair/in_conversation_cluster",
            "Armchair is in seating cluster",
            weight,
            0,
            "skipped",
            "No armchair found.",
        )
    if sofa is None:
        return RuleResult(
            "armchair/in_conversation_cluster",
            "Armchair is in seating cluster",
            weight,
            0,
            "skipped",
            "No sofa found.",
        )

    d_sofa = _euclidean(_bbox_center(armchair.bbox), _bbox_center(sofa.bbox))
    if coffee_table is not None:
        d_table = _euclidean(
            _bbox_center(armchair.bbox), _bbox_center(coffee_table.bbox)
        )
        passed = (
            d_sofa <= cfg.seat_conversation_max_center_distance_mm
            and d_table <= cfg.armchair_near_coffee_table_mm
        )
        details = f"dist_to_sofa_mm={round(d_sofa, 1)}; dist_to_coffee_table_mm={round(d_table, 1)}."
    else:
        passed = d_sofa <= cfg.seat_conversation_max_center_distance_mm
        details = f"dist_to_sofa_mm={round(d_sofa, 1)}."
    return RuleResult(
        "armchair/in_conversation_cluster",
        "Armchair is in seating cluster",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        details,
    )


def _rule_armchair_typical_position(
    armchair: StyledObject | None,
    sofa: StyledObject | None,
    coffee_table: StyledObject | None,
    tv_console: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 7
    if armchair is None:
        return RuleResult(
            "armchair/diagonal_to_coffee_table",
            "Armchair placed near conversation area",
            weight,
            0,
            "skipped",
            "No armchair found.",
        )
    if sofa is None or coffee_table is None:
        return RuleResult(
            "armchair/diagonal_to_coffee_table",
            "Armchair placed near conversation area",
            weight,
            0,
            "skipped",
            "Missing sofa or coffee_table.",
        )

    arm_center = _bbox_center(armchair.bbox)
    sofa_center = _bbox_center(sofa.bbox)
    table_center = _bbox_center(coffee_table.bbox)
    d_table = _euclidean(arm_center, table_center)
    d_sofa = _euclidean(arm_center, sofa_center)
    ok = (
        d_table <= cfg.armchair_near_coffee_table_mm
        and d_sofa <= cfg.seat_conversation_max_center_distance_mm
    )
    if tv_console is not None:
        d_tv = _euclidean(arm_center, _bbox_center(tv_console.bbox))
        ok = ok and d_tv >= 700
        details = f"dist_to_table_mm={round(d_table, 1)}; dist_to_sofa_mm={round(d_sofa, 1)}; dist_to_tv_console_mm={round(d_tv, 1)}."
    else:
        details = (
            f"dist_to_table_mm={round(d_table, 1)}; dist_to_sofa_mm={round(d_sofa, 1)}."
        )
    return RuleResult(
        "armchair/diagonal_to_coffee_table",
        "Armchair placed near conversation area",
        weight,
        (weight if ok else 0),
        ("passed" if ok else "failed"),
        details,
    )


def _rule_object_not_block_path_optional(
    rule_id: str,
    name: str,
    weight: float,
    obj: StyledObject | None,
    door_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    room_centroid: tuple[float, float],
) -> RuleResult:
    if obj is None:
        return RuleResult(rule_id, name, weight, 0, "skipped", "Object not present.")
    if not door_segments:
        return RuleResult(
            rule_id, name, weight, 0, "skipped", "No door opening segment found."
        )

    door_mid = _segment_midpoint(door_segments[0][1], door_segments[0][2])
    blocked = _segment_intersects_rect(door_mid, room_centroid, obj.bbox)
    return RuleResult(
        rule_id,
        name,
        weight,
        (weight if not blocked else 0),
        ("passed" if not blocked else "failed"),
        f"door_to_centroid_intersects={blocked}.",
    )


def _rule_side_table_next_to_seat(
    side_table: StyledObject | None,
    sofa: StyledObject | None,
    armchair: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if side_table is None:
        return RuleResult(
            "side_table/next_to_seating",
            "Side table next to sofa or armchair",
            weight,
            0,
            "skipped",
            "No side_table found.",
        )
    if sofa is None and armchair is None:
        return RuleResult(
            "side_table/next_to_seating",
            "Side table next to sofa or armchair",
            weight,
            0,
            "skipped",
            "No sofa or armchair found.",
        )

    targets: list[StyledObject] = []
    if sofa is not None:
        targets.append(sofa)
    if armchair is not None:
        targets.append(armchair)

    d = min(
        _euclidean(_bbox_center(side_table.bbox), _bbox_center(t.bbox)) for t in targets
    )
    passed = d <= cfg.side_table_near_seat_mm
    return RuleResult(
        "side_table/next_to_seating",
        "Side table next to sofa or armchair",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"min_dist_to_seat_mm={round(d, 1)} (max={cfg.side_table_near_seat_mm}).",
    )


def _rule_side_table_not_replacing_coffee_table(
    side_table: StyledObject | None,
    sofa: StyledObject | None,
    armchair: StyledObject | None,
    coffee_table: StyledObject | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 7
    if side_table is None:
        return RuleResult(
            "side_table/not_replacing_coffee_table",
            "Side table is not used as coffee table",
            weight,
            0,
            "skipped",
            "No side_table found.",
        )
    if coffee_table is None:
        return RuleResult(
            "side_table/not_replacing_coffee_table",
            "Side table is not used as coffee table",
            weight,
            0,
            "skipped",
            "No coffee_table found.",
        )
    if sofa is None and armchair is None:
        return RuleResult(
            "side_table/not_replacing_coffee_table",
            "Side table is not used as coffee table",
            weight,
            0,
            "skipped",
            "No sofa/armchair found.",
        )

    side_center = _bbox_center(side_table.bbox)
    coffee_center = _bbox_center(coffee_table.bbox)

    targets: list[StyledObject] = []
    if sofa is not None:
        targets.append(sofa)
    if armchair is not None:
        targets.append(armchair)
    dist_to_seat = min(_euclidean(side_center, _bbox_center(t.bbox)) for t in targets)
    dist_to_coffee = _euclidean(side_center, coffee_center)

    passed = (
        dist_to_seat <= cfg.side_table_near_seat_mm and dist_to_seat <= dist_to_coffee
    )
    return RuleResult(
        "side_table/not_replacing_coffee_table",
        "Side table is not used as coffee table",
        weight,
        (weight if passed else 0),
        ("passed" if passed else "failed"),
        f"dist_to_seat_mm={round(dist_to_seat, 1)}; dist_to_coffee_table_mm={round(dist_to_coffee, 1)}.",
    )


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.min_x + bbox.max_x) / 2.0, (bbox.min_y + bbox.max_y) / 2.0)


def _inflate_bbox(bbox: BBox, pad: int) -> BBox:
    p = max(0, int(pad))
    return BBox(
        min_x=bbox.min_x - p,
        min_y=bbox.min_y - p,
        max_x=bbox.max_x + p,
        max_y=bbox.max_y + p,
    )


@dataclass(frozen=True)
class _RectEdge:
    side: Literal["top", "bottom", "left", "right"]
    a: tuple[float, float]
    b: tuple[float, float]

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.a[0] + self.b[0]) / 2.0, (self.a[1] + self.b[1]) / 2.0)


def _rect_edges(bbox: BBox) -> list[_RectEdge]:
    min_x, min_y, max_x, max_y = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
    return [
        _RectEdge(side="top", a=(min_x, min_y), b=(max_x, min_y)),
        _RectEdge(side="bottom", a=(min_x, max_y), b=(max_x, max_y)),
        _RectEdge(side="left", a=(min_x, min_y), b=(min_x, max_y)),
        _RectEdge(side="right", a=(max_x, min_y), b=(max_x, max_y)),
    ]


def _infer_object_front_direction_from_wall(
    bbox: BBox, room_poly: list[Point]
) -> tuple[float, float] | None:
    if len(room_poly) < 3:
        return None
    edges = _rect_edges(bbox)
    back = min(
        edges, key=lambda e: _min_distance_point_to_polygon_edges(e.midpoint, room_poly)
    )
    if back.side == "top":
        return (0.0, 1.0)
    if back.side == "bottom":
        return (0.0, -1.0)
    if back.side == "left":
        return (1.0, 0.0)
    return (-1.0, 0.0)


def _segment_midpoint(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _obstacle_bboxes(obstacles: list[dict]) -> dict[str, list[BBox]]:
    out: dict[str, list[BBox]] = {}
    for ob in obstacles:
        if not isinstance(ob, dict):
            continue
        ob_type = ob.get("type")
        if not isinstance(ob_type, str) or not ob_type:
            continue
        poly = ob.get("polygon_ccw")
        if not isinstance(poly, list) or not poly:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for p in poly:
            if not isinstance(p, dict):
                continue
            x = p.get("x")
            y = p.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            xs.append(float(x))
            ys.append(float(y))
        if not xs or not ys:
            continue
        bbox = BBox(
            min_x=int(min(xs)),
            min_y=int(min(ys)),
            max_x=int(max(xs)),
            max_y=int(max(ys)),
        )
        out.setdefault(ob_type, []).append(bbox)
    return out


def _opening_segments(
    openings: list[dict],
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for o in openings:
        if not isinstance(o, dict):
            continue
        oid = o.get("id")
        seg = o.get("segment_mm")
        if not isinstance(oid, str) or not oid:
            continue
        if not isinstance(seg, list) or len(seg) != 2:
            continue
        a, b = seg
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        ax, ay = a.get("x"), a.get("y")
        bx, by = b.get("x"), b.get("y")
        if not all(isinstance(v, (int, float)) for v in (ax, ay, bx, by)):
            continue
        segments.append((oid, (float(ax), float(ay)), (float(bx), float(by))))
    return segments


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _avg_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _distance_point_to_segment(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 <= 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def _polygon_edges(
    poly: list[Point],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = [(float(p.x), float(p.y)) for p in poly]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _min_distance_point_to_polygon_edges(
    p: tuple[float, float], poly: list[Point]
) -> float:
    return min(_distance_point_to_segment(p, a, b) for a, b in _polygon_edges(poly))


def _bbox_sample_points(bbox: BBox) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (mid_x, min_y),
        (mid_x, max_y),
        (min_x, mid_y),
        (max_x, mid_y),
        (mid_x, mid_y),
    ]


def _min_distance_bbox_to_polygon(bbox: BBox, poly: list[Point]) -> float:
    points = _bbox_sample_points(bbox)
    return min(_min_distance_point_to_polygon_edges(p, poly) for p in points)


def _segment_intersects_rect(
    a: tuple[float, float], b: tuple[float, float], r: BBox
) -> bool:
    seg_min_x, seg_max_x = min(a[0], b[0]), max(a[0], b[0])
    seg_min_y, seg_max_y = min(a[1], b[1]), max(a[1], b[1])
    if (
        seg_max_x < r.min_x
        or seg_min_x > r.max_x
        or seg_max_y < r.min_y
        or seg_min_y > r.max_y
    ):
        return False
    if _point_in_bbox(a, r) or _point_in_bbox(b, r):
        return True
    edges = [
        ((r.min_x, r.min_y), (r.max_x, r.min_y)),
        ((r.max_x, r.min_y), (r.max_x, r.max_y)),
        ((r.max_x, r.max_y), (r.min_x, r.max_y)),
        ((r.min_x, r.max_y), (r.min_x, r.min_y)),
    ]
    return any(_segments_intersect(a, b, e1, e2) for e1, e2 in edges)


def _point_in_bbox(p: tuple[float, float], r: BBox) -> bool:
    return r.min_x <= p[0] <= r.max_x and r.min_y <= p[1] <= r.max_y


def _point_in_bbox_padded(
    p: tuple[float, float], r: BBox, *, padding_mm: int = 0
) -> bool:
    return (
        r.min_x - padding_mm <= p[0] <= r.max_x + padding_mm
        and r.min_y - padding_mm <= p[1] <= r.max_y + padding_mm
    )


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    def orient(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
            1
        ] <= max(p[1], r[1])

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if o1 == 0 and on_segment(a1, b1, a2):
        return True
    if o2 == 0 and on_segment(a1, b2, a2):
        return True
    if o3 == 0 and on_segment(b1, a1, b2):
        return True
    if o4 == 0 and on_segment(b1, a2, b2):
        return True
    return False


def _rects_intersect(a: BBox, b: BBox) -> bool:
    return not (
        a.max_x <= b.min_x
        or a.min_x >= b.max_x
        or a.max_y <= b.min_y
        or a.min_y >= b.max_y
    )


def _bbox_union(bboxes: list[BBox]) -> BBox:
    xs = [b.min_x for b in bboxes] + [b.max_x for b in bboxes]
    ys = [b.min_y for b in bboxes] + [b.max_y for b in bboxes]
    return BBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _intersection_area_ratio(a: BBox, b: BBox) -> float:
    ix0 = max(a.min_x, b.min_x)
    iy0 = max(a.min_y, b.min_y)
    ix1 = min(a.max_x, b.max_x)
    iy1 = min(a.max_y, b.max_y)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_b = max(1, (b.max_x - b.min_x) * (b.max_y - b.min_y))
    return inter / area_b


def _min_gap_between_bboxes(a: BBox, b: BBox) -> float:
    dx = max(b.min_x - a.max_x, a.min_x - b.max_x, 0)
    dy = max(b.min_y - a.max_y, a.min_y - b.max_y, 0)
    if dx == 0 and dy == 0:
        return 0.0
    return math.hypot(dx, dy)


def _axis_overlap_ratio(a0: int, a1: int, b0: int, b1: int) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    overlap = max(0, hi - lo)
    denom = max(1, min(abs(a1 - a0), abs(b1 - b0)))
    return overlap / denom


def _max_axis_overlap_ratio(a: BBox, b: BBox) -> float:
    return max(
        _axis_overlap_ratio(a.min_x, a.max_x, b.min_x, b.max_x),
        _axis_overlap_ratio(a.min_y, a.max_y, b.min_y, b.max_y),
    )


def _polygon_centroid(poly: list[Point]) -> tuple[float, float]:
    pts = [(float(p.x), float(p.y)) for p in poly]
    if len(pts) < 3:
        return (0.0, 0.0)
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if a == 0:
        return _avg_point(pts)
    a *= 0.5
    cx /= 6.0 * a
    cy /= 6.0 * a
    return (cx, cy)


def _room_diag(room_poly: list[Point]) -> float:
    if len(room_poly) < 3:
        return 0.0
    xs = [p.x for p in room_poly]
    ys = [p.y for p in room_poly]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _living_room_key_objects(
    objects_by_cat: dict[str, list[StyledObject]],
) -> list[StyledObject]:
    out: list[StyledObject] = []
    for cat in ("sofa", "coffee_table", "tv_console", "tv", "armchair", "side_table"):
        out.extend(objects_by_cat.get(cat, []))
    return out


def _oob_skipped(details: str) -> OobResult:
    return OobResult(
        status="skipped",
        scene_invalid=None,
        oob_rate=None,
        checked_instance_ids=[],
        out_of_bounds_instance_ids=[],
        out_of_bounds_objects=[],
        details=details,
    )


def _compute_oob(room_poly: list[Point], objects: list[StyledObject]) -> OobResult:
    checked = [o.instance_id for o in objects]
    if not checked:
        return _oob_skipped("No evaluated objects found.")
    if len(room_poly) < 3:
        return _oob_skipped("Room polygon missing/invalid.")

    out_of_bounds_objects = [
        {"instance_id": o.instance_id, "object_type": o.object_type}
        for o in objects
        if not _bbox_within_polygon(o.bbox, room_poly)
    ]
    out_of_bounds = [o["instance_id"] for o in out_of_bounds_objects]
    invalid = bool(out_of_bounds)
    return OobResult(
        status=("failed" if invalid else "passed"),
        scene_invalid=invalid,
        oob_rate=(1.0 if invalid else 0.0),
        checked_instance_ids=checked,
        out_of_bounds_instance_ids=out_of_bounds,
        out_of_bounds_objects=out_of_bounds_objects,
        details=f"out_of_bounds={len(out_of_bounds)}/{len(checked)}.",
    )


def _compute_bbl(objects: list[StyledObject]) -> BblResult:
    checked = [o.instance_id for o in objects]
    if len(objects) < 2:
        return BblResult(
            status="skipped",
            mean_intersection_area_mm2=None,
            pair_count=0,
            checked_instance_ids=checked,
            details="Need at least 2 evaluated objects.",
        )

    total = 0.0
    pair_count = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            total += float(_bbox_intersection_area(objects[i].bbox, objects[j].bbox))
            pair_count += 1

    mean = (total / pair_count) if pair_count > 0 else 0.0
    return BblResult(
        status="passed",
        mean_intersection_area_mm2=round(mean, 3),
        pair_count=pair_count,
        checked_instance_ids=checked,
        details=f"sum_intersection_area_mm2={round(total, 3)}.",
    )


def _compute_numerical(
    objects_by_cat: dict[str, list[StyledObject]],
    spec: dict[str, tuple[int, int]],
    comparisons: list[tuple[str, str]],
) -> NumericalResult:
    pred_counts_raw = {cat: len(objects_by_cat.get(cat, [])) for cat in spec}
    pred_counts = {cat: (1 if pred_counts_raw.get(cat, 0) > 0 else 0) for cat in spec}
    gt_spec = {
        cat: {"min": min_c, "max": max_c} for cat, (min_c, max_c) in spec.items()
    }

    tp = 0
    fp = 0
    fn = 0
    gt_effective: dict[str, int] = {}
    for cat, (min_c, max_c) in spec.items():
        pred = pred_counts.get(cat, 0)
        gt_eff = int(min(max(pred, min_c), max_c))
        gt_effective[cat] = gt_eff
        tp += min(pred, gt_eff)
        fp += max(pred - gt_eff, 0)
        fn += max(gt_eff - pred, 0)

    pred_total = sum(pred_counts.values())
    gt_total = sum(gt_effective.values())
    precision = (tp / pred_total) if pred_total > 0 else None
    recall = (tp / gt_total) if gt_total > 0 else None
    accuracy = (
        1.0
        if all(
            min_c <= pred_counts[cat] <= max_c for cat, (min_c, max_c) in spec.items()
        )
        else 0.0
    )

    comparison_results: list[ComparisonResult] = []
    for left, right in comparisons:
        pred_rel = _compare_counts(pred_counts.get(left, 0), pred_counts.get(right, 0))
        allowed = _allowed_relations(spec.get(left, (0, 0)), spec.get(right, (0, 0)))
        comparison_results.append(
            ComparisonResult(
                left=left,
                right=right,
                predicted_relation=pred_rel,
                allowed_relations=_relation_list(allowed),
                correct=(pred_rel in allowed),
            )
        )
    comparison_accuracy = (
        (sum(1 for c in comparison_results if c.correct) / len(comparison_results))
        if comparison_results
        else None
    )

    return NumericalResult(
        status="passed",
        gt_spec=gt_spec,
        pred_counts=pred_counts,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=(round(precision, 6) if precision is not None else None),
        recall=(round(recall, 6) if recall is not None else None),
        accuracy=accuracy,
        comparison_accuracy=(
            round(comparison_accuracy, 6) if comparison_accuracy is not None else None
        ),
        comparisons=comparison_results,
        details=f"pred_total={pred_total}; gt_total={gt_total}.",
    )


def _relation_list(relations: set[Relation]) -> list[Relation]:
    order: list[Relation] = ["less", "equal", "greater"]
    return [r for r in order if r in relations]


def _compare_counts(a: int, b: int) -> Relation:
    if a < b:
        return "less"
    if a > b:
        return "greater"
    return "equal"


def _allowed_relations(
    a_range: tuple[int, int], b_range: tuple[int, int]
) -> set[Relation]:
    a_min, a_max = a_range
    b_min, b_max = b_range
    allowed: set[Relation] = set()
    if a_min < b_max:
        allowed.add("less")
    if a_max > b_min:
        allowed.add("greater")
    if max(a_min, b_min) <= min(a_max, b_max):
        allowed.add("equal")
    return allowed


def _bbox_intersection_area(a: BBox, b: BBox) -> int:
    ix = max(0, min(a.max_x, b.max_x) - max(a.min_x, b.min_x))
    iy = max(0, min(a.max_y, b.max_y) - max(a.min_y, b.min_y))
    return int(ix * iy)


def _bbox_within_polygon(bbox: BBox, room_poly: list[Point]) -> bool:
    return all(
        _point_in_polygon_or_on_edge(p, room_poly) for p in _bbox_sample_points(bbox)
    )


def _point_in_polygon_or_on_edge(p: tuple[float, float], poly: list[Point]) -> bool:
    x, y = p
    pts = [(float(pt.x), float(pt.y)) for pt in poly]
    if len(pts) < 3:
        return False
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        if _point_on_segment((x, y), a, b):
            return True

    inside = False
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x_int > x:
                inside = not inside
    return inside


def _point_on_segment(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> bool:
    px, py = p
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-6:
        return False
    return (min(ax, bx) - 1e-6) <= px <= (max(ax, bx) + 1e-6) and (
        min(ay, by) - 1e-6
    ) <= py <= (max(ay, by) + 1e-6)
