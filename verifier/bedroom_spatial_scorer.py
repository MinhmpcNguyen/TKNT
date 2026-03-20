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
    bed_center_max_ratio_of_room_diag: float = 0.45
    headboard_against_wall_mm: int = 200
    nightstand_adjacent_gap_mm: int = 300
    nightstand_overlap_tol_mm: int = 50
    nightstand_side_clearance_diff_mm: int = 150
    bedside_lamp_association_mm: int = 450
    bedside_lamp_near_bed_mm: int = 2000
    bedside_lamp_near_nightstand_mm: int = 800
    wardrobe_front_clearance_mm: int = 800
    wardrobe_far_from_headboard_mm: int = 800
    bed_avoid_window_headboard_mm: int = 500
    side_table_near_window_mm: int = 1500
    side_table_near_bed_mm: int = 800
    side_table_opposite_wall_mm: int = 450
    ceiling_light_near_object_max_mm: int = 500


RuleStatus = Literal["passed", "failed", "skipped"]
ScoreStatus = Literal["OK", "NOT_APPLICABLE", "NEED_INFO", "ERROR"]
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
    "bed": {"bed"},
    "nightstand": {"nightstand", "bedside_table"},
    "wardrobe": {"wardrobe", "closet"},
    "curtain": {"curtain", "blinds", "curtain_or_blinds"},
    "ceiling_lamp": {
        "ceiling_lamp",
        "ceiling_light",
        "celling_lamp",
        "celling_light",
        "chandelier",
        "pendant_light",
    },
    "bedside_lamp": {"bedside_lamp", "table_lamp", "night_lamp"},
    "side_table": {"side_table", "desk", "small_table", "writing_desk"},
}

_BEDROOM_NUMERICAL_SPEC: dict[str, tuple[int, int]] = {
    "bed": (1, 1),
    "nightstand": (1, 2),
    "wardrobe": (1, 1),
    "curtain": (1, 1),
    "ceiling_lamp": (1, 1),
    "bedside_lamp": (1, 1),
    "side_table": (1, 1),
}

_BEDROOM_COMPARISONS: list[tuple[str, str]] = [
    ("nightstand", "bed"),
    ("wardrobe", "bed"),
]


def score_bedroom_spatial(
    stylist_output: StylistOutput,
    *,
    config: ScoringConfig | None = None,
) -> ScoreResult:
    cfg = config or ScoringConfig()

    objects_by_cat = _objects_by_category(stylist_output.objects)
    bedroom_key_objects = _bedroom_key_objects(objects_by_cat)
    room_poly = stylist_output.room.polygon_ccw if stylist_output.room else []

    oob = (
        _compute_oob(room_poly, stylist_output.objects)
        if stylist_output.room
        else _oob_skipped("Room missing.")
    )
    bbl = _compute_bbl(bedroom_key_objects)
    numerical = _compute_numerical(
        objects_by_cat, _BEDROOM_NUMERICAL_SPEC, _BEDROOM_COMPARISONS
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
    if room_type != "bedroom":
        return ScoreResult(
            status="NOT_APPLICABLE",
            room_type=stylist_output.room.room_type,
            overall_score=0.0,
            max_score=0.0,
            normalized_score_0_100=None,
            rules=[],
            object_instances=object_instances,
            oob=_oob_skipped("Not applicable (room_type is not bedroom)."),
            bbl=bbl,
            numerical=numerical,
        )

    bed = _primary(objects_by_cat.get("bed", []))
    nightstands = objects_by_cat.get("nightstand", [])
    bedside_lamps = objects_by_cat.get("bedside_lamp", [])
    wardrobes = objects_by_cat.get("wardrobe", [])
    curtains = objects_by_cat.get("curtain", [])
    ceiling_lights = objects_by_cat.get("ceiling_lamp", [])
    side_tables = objects_by_cat.get("side_table", [])

    door_segments = _opening_segments(
        (stylist_output.room.openings.doors if stylist_output.room.openings else [])
    )
    window_segments = _opening_segments(
        (stylist_output.room.openings.windows if stylist_output.room.openings else [])
    )

    rules: list[RuleResult] = []

    # Bed (27)
    rules.append(_rule_bed_central(bed, room_poly, cfg))
    rules.append(_rule_bed_against_wall(bed, room_poly, cfg))
    headboard_edge = (
        _infer_bed_headboard_edge(bed, nightstands, room_poly, cfg) if bed else None
    )
    rules.append(_rule_bed_headboard_against_wall(bed, headboard_edge, room_poly, cfg))

    # Nightstand (16)
    rules.append(
        _rule_nightstands_adjacent_to_headboard(bed, headboard_edge, nightstands, cfg)
    )
    rules.append(
        _rule_single_nightstand_preferred_side(
            bed, headboard_edge, nightstands, room_poly, cfg
        )
    )

    # Bedside lamp (26)
    rules.append(_rule_nightstand_with_bedside_lamp(nightstands, bedside_lamps, cfg))
    rules.append(
        _rule_bedside_lamp_on_or_above_nightstand(nightstands, bedside_lamps, cfg)
    )
    rules.append(
        _rule_bedside_lamp_not_independent(bed, nightstands, bedside_lamps, cfg)
    )

    # Wardrobe (34)
    wardrobe = _primary(wardrobes)
    rules.append(_rule_wardrobe_against_wall(wardrobe, room_poly, cfg))
    rules.append(
        _rule_wardrobe_near_door_than_bed(wardrobe, bed, headboard_edge, door_segments)
    )
    rules.append(
        _rule_wardrobe_not_next_to_headboard(wardrobe, bed, headboard_edge, cfg)
    )
    rules.append(
        _rule_wardrobe_front_clearance(wardrobe, stylist_output.objects, room_poly, cfg)
    )

    # Side table (22)
    side_table = _primary(side_tables)
    rules.append(_rule_side_table_near_window(side_table, window_segments, cfg))
    rules.append(
        _rule_side_table_alternative_placement(
            side_table, bed, headboard_edge, room_poly, window_segments, cfg
        )
    )
    rules.append(
        _rule_side_table_not_blocking_paths(side_table, bed, wardrobe, door_segments)
    )

    # Curtain / window (18)
    rules.append(_rule_curtain_present_with_window(curtains, window_segments))
    rules.append(_rule_bed_not_under_window(bed, headboard_edge, window_segments, cfg))

    # Ceiling light (17)
    rules.append(_rule_ceiling_light_near_center(ceiling_lights, room_poly))
    rules.append(
        _rule_ceiling_light_general_overhead(
            ceiling_lights, [bed, wardrobe, side_table], cfg
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


def _rule_bed_present(bed: StyledObject | None) -> RuleResult:
    if bed is None:
        return RuleResult(
            rule_id="bed/presence",
            name="Bed present",
            weight=10,
            earned=0,
            status="failed",
            details="No bed found (object_type=bed).",
        )
    return RuleResult(
        rule_id="bed/presence",
        name="Bed present",
        weight=10,
        earned=10,
        status="passed",
        details=f"Found bed: {bed.instance_id}.",
    )


def _rule_bed_central(
    bed: StyledObject | None,
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 10
    if bed is None:
        return RuleResult(
            rule_id="bed/central",
            name="Bed is central anchor",
            weight=weight,
            earned=0,
            status="failed",
            details="No bed found.",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="bed/central",
            name="Bed is central anchor",
            weight=weight,
            earned=0,
            status="skipped",
            details="Room polygon missing/invalid.",
        )

    room_center = _polygon_centroid(room_poly)
    room_bbox = _polygon_bbox(room_poly)
    diag = math.hypot(
        room_bbox.max_x - room_bbox.min_x, room_bbox.max_y - room_bbox.min_y
    )
    if diag <= 0:
        diag = 1.0
    d = _euclidean(_bbox_center(bed.bbox), room_center)
    threshold = cfg.bed_center_max_ratio_of_room_diag * diag
    passed = d <= threshold
    return RuleResult(
        rule_id="bed/central",
        name="Bed is central anchor",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"dist_to_center_mm={round(d, 1)} (threshold={round(threshold, 1)}).",
    )


def _rule_bed_against_wall(
    bed: StyledObject | None,
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    if bed is None:
        return RuleResult(
            rule_id="bed/against_wall",
            name="Bed near a wall",
            weight=8,
            earned=0,
            status="failed",
            details="No bed found.",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="bed/against_wall",
            name="Bed near a wall",
            weight=8,
            earned=0,
            status="skipped",
            details="Room polygon missing/invalid.",
        )

    d = _min_distance_bbox_to_polygon(bed.bbox, room_poly)
    passed = d <= cfg.object_against_wall_mm
    return RuleResult(
        rule_id="bed/against_wall",
        name="Bed near a wall",
        weight=8,
        earned=(8 if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"min_distance_to_wall_mm={round(d, 1)} (threshold={cfg.object_against_wall_mm}).",
    )


@dataclass(frozen=True)
class _RectEdge:
    side: Literal["top", "bottom", "left", "right"]
    a: tuple[float, float]
    b: tuple[float, float]

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.a[0] + self.b[0]) / 2.0, (self.a[1] + self.b[1]) / 2.0)


def _infer_bed_headboard_edge(
    bed: StyledObject | None,
    nightstands: list[StyledObject],
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> _RectEdge | None:
    if bed is None:
        return None

    edges = _rect_edges(bed.bbox)
    if nightstands:
        edge_scored: list[tuple[int, float, _RectEdge]] = []
        for e in edges:
            adjacent_count = sum(
                1
                for ns in nightstands
                if _is_rect_adjacent_to_bed_edge(ns.bbox, bed.bbox, e, cfg)
            )
            wall_dist = (
                _min_distance_point_to_polygon_edges(e.midpoint, room_poly)
                if len(room_poly) >= 3
                else float("inf")
            )
            edge_scored.append((adjacent_count, wall_dist, e))

        best_adj = max(edge_scored, key=lambda t: (t[0], -t[1]))[0]
        best = [t for t in edge_scored if t[0] == best_adj]
        _, _, chosen = min(best, key=lambda t: t[1])
        return chosen

    if len(room_poly) >= 3:
        return min(
            edges,
            key=lambda e: _min_distance_point_to_polygon_edges(e.midpoint, room_poly),
        )
    return edges[0]


def _rect_edges(bbox: BBox) -> list[_RectEdge]:
    min_x, min_y, max_x, max_y = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
    return [
        _RectEdge(side="top", a=(min_x, min_y), b=(max_x, min_y)),
        _RectEdge(side="bottom", a=(min_x, max_y), b=(max_x, max_y)),
        _RectEdge(side="left", a=(min_x, min_y), b=(min_x, max_y)),
        _RectEdge(side="right", a=(max_x, min_y), b=(max_x, max_y)),
    ]


def _rule_bed_headboard_against_wall(
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    if bed is None or headboard_edge is None:
        return RuleResult(
            rule_id="bed/headboard_against_wall",
            name="Headboard near a wall",
            weight=9,
            earned=0,
            status="failed",
            details="No bed found (or cannot infer headboard edge).",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="bed/headboard_against_wall",
            name="Headboard near a wall",
            weight=9,
            earned=0,
            status="skipped",
            details="Room polygon missing/invalid.",
        )

    d = _min_distance_point_to_polygon_edges(headboard_edge.midpoint, room_poly)
    passed = d <= cfg.headboard_against_wall_mm
    return RuleResult(
        rule_id="bed/headboard_against_wall",
        name="Headboard near a wall",
        weight=9,
        earned=(9 if passed else 0),
        status=("passed" if passed else "failed"),
        details=(
            f"headboard_edge={headboard_edge.side}; min_distance_to_wall_mm={round(d, 1)} "
            f"(threshold={cfg.headboard_against_wall_mm})."
        ),
    )


def _rule_nightstands_adjacent_to_headboard(
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    nightstands: list[StyledObject],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 10
    if bed is None or headboard_edge is None:
        return RuleResult(
            rule_id="nightstand/adjacent_to_headboard",
            name="Nightstand(s) adjacent to headboard",
            weight=weight,
            earned=0,
            status="failed",
            details="No bed found (or cannot infer headboard edge).",
        )

    if not nightstands:
        return RuleResult(
            rule_id="nightstand/adjacent_to_headboard",
            name="Nightstand(s) adjacent to headboard",
            weight=weight,
            earned=0,
            status="failed",
            details="No nightstand found.",
        )

    adjacent = [
        ns
        for ns in nightstands
        if _is_rect_adjacent_to_bed_edge(ns.bbox, bed.bbox, headboard_edge, cfg)
    ]
    expected = min(2, len(nightstands))
    earned = weight * (len(adjacent) / expected) if expected > 0 else 0
    passed = len(adjacent) >= 1
    return RuleResult(
        rule_id="nightstand/adjacent_to_headboard",
        name="Nightstand(s) adjacent to headboard",
        weight=weight,
        earned=round(earned, 3),
        status=("passed" if passed else "failed"),
        details=f"adjacent={len(adjacent)}/{expected}; headboard_edge={headboard_edge.side}.",
    )


def _rule_single_nightstand_preferred_side(
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    nightstands: list[StyledObject],
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 6
    if len(nightstands) != 1:
        return RuleResult(
            rule_id="nightstand/preferred_side",
            name="Single nightstand on preferred side",
            weight=weight,
            earned=0,
            status="skipped",
            details="Single-side rule only applies when nightstand count is exactly 1.",
        )
    if bed is None or headboard_edge is None:
        return RuleResult(
            rule_id="nightstand/preferred_side",
            name="Single nightstand on preferred side",
            weight=weight,
            earned=0,
            status="failed",
            details="No bed found (or cannot infer headboard edge).",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="nightstand/preferred_side",
            name="Single nightstand on preferred side",
            weight=weight,
            earned=0,
            status="skipped",
            details="Room polygon missing/invalid.",
        )

    preferred = _preferred_single_nightstand_side(
        bed.bbox, headboard_edge, room_poly, cfg
    )
    ns = nightstands[0]
    actual = _nightstand_side_relative_to_bed(ns.bbox, bed.bbox, headboard_edge)

    if preferred == "either":
        passed = True
    elif preferred is None or actual is None:
        passed = False
    else:
        passed = preferred == actual

    return RuleResult(
        rule_id="nightstand/preferred_side",
        name="Single nightstand on preferred side",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"nightstand={ns.instance_id}; preferred={preferred}; actual={actual}.",
    )


def _rule_nightstand_with_bedside_lamp(
    nightstands: list[StyledObject],
    bedside_lamps: list[StyledObject],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if not nightstands:
        return RuleResult(
            rule_id="nightstand/paired_with_bedside_lamp",
            name="Nightstand paired with bedside lamp",
            weight=weight,
            earned=0,
            status="failed",
            details="No nightstand found.",
        )
    if not bedside_lamps:
        return RuleResult(
            rule_id="nightstand/paired_with_bedside_lamp",
            name="Nightstand paired with bedside lamp",
            weight=weight,
            earned=0,
            status="failed",
            details="No bedside_lamp found.",
        )

    ns_ids = {n.instance_id for n in nightstands}
    paired = [
        lamp
        for lamp in bedside_lamps
        if _lamp_associated_with_nightstand(
            lamp, nightstands, ns_ids, cfg.bedside_lamp_association_mm
        )
    ]
    passed = bool(paired)
    return RuleResult(
        rule_id="nightstand/paired_with_bedside_lamp",
        name="Nightstand paired with bedside lamp",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=(
            f"paired_bedside_lamps={len(paired)}; "
            f"nightstands={len(nightstands)}; bedside_lamps={len(bedside_lamps)}."
        ),
    )


def _preferred_single_nightstand_side(
    bed_bbox: BBox,
    headboard_edge: _RectEdge,
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> str | None:
    if len(room_poly) < 3:
        return None

    bed_mid_x = (bed_bbox.min_x + bed_bbox.max_x) / 2.0
    bed_mid_y = (bed_bbox.min_y + bed_bbox.max_y) / 2.0

    if headboard_edge.side in {"top", "bottom"}:
        left_mid = (bed_bbox.min_x, bed_mid_y)
        right_mid = (bed_bbox.max_x, bed_mid_y)
        left_clear = _min_distance_point_to_polygon_edges(left_mid, room_poly)
        right_clear = _min_distance_point_to_polygon_edges(right_mid, room_poly)
        diff = abs(left_clear - right_clear)
        if diff <= cfg.nightstand_side_clearance_diff_mm:
            return "either"
        return "left" if left_clear > right_clear else "right"

    if headboard_edge.side in {"left", "right"}:
        bottom_mid = (bed_mid_x, bed_bbox.min_y)
        top_mid = (bed_mid_x, bed_bbox.max_y)
        bottom_clear = _min_distance_point_to_polygon_edges(bottom_mid, room_poly)
        top_clear = _min_distance_point_to_polygon_edges(top_mid, room_poly)
        diff = abs(bottom_clear - top_clear)
        if diff <= cfg.nightstand_side_clearance_diff_mm:
            return "either"
        return "bottom" if bottom_clear > top_clear else "top"

    return None


def _nightstand_side_relative_to_bed(
    nightstand_bbox: BBox,
    bed_bbox: BBox,
    headboard_edge: _RectEdge,
) -> str | None:
    ns_center = _bbox_center(nightstand_bbox)
    bed_center = _bbox_center(bed_bbox)
    if headboard_edge.side in {"top", "bottom"}:
        return "left" if ns_center[0] < bed_center[0] else "right"
    if headboard_edge.side in {"left", "right"}:
        return "bottom" if ns_center[1] < bed_center[1] else "top"
    return None


def _rule_bedside_lamp_on_or_above_nightstand(
    nightstands: list[StyledObject],
    bedside_lamps: list[StyledObject],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if not bedside_lamps:
        return RuleResult(
            rule_id="bedside_lamp/on_nightstand_or_above",
            name="Bedside lamp on/above nightstand",
            weight=weight,
            earned=0,
            status="failed",
            details="No bedside_lamp found.",
        )
    if not nightstands:
        return RuleResult(
            rule_id="bedside_lamp/on_nightstand_or_above",
            name="Bedside lamp on/above nightstand",
            weight=weight,
            earned=0,
            status="failed",
            details="No nightstand found.",
        )

    ns_ids = {n.instance_id for n in nightstands}
    associated = [
        lamp
        for lamp in bedside_lamps
        if _lamp_associated_with_nightstand(
            lamp, nightstands, ns_ids, cfg.bedside_lamp_association_mm
        )
    ]
    passed = bool(associated)
    return RuleResult(
        rule_id="bedside_lamp/on_nightstand_or_above",
        name="Bedside lamp on/above nightstand",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"associated_with_nightstand={len(associated)}; bedside_lamps={len(bedside_lamps)}.",
    )


def _lamp_associated_with_nightstand(
    lamp: StyledObject,
    nightstands: list[StyledObject],
    nightstand_ids: set[str],
    max_center_distance_mm: int,
) -> bool:
    if lamp.place_on is not None and lamp.place_on.target_instance_id in nightstand_ids:
        return True
    lamp_center = _bbox_center(lamp.bbox)
    for ns in nightstands:
        if _euclidean(lamp_center, _bbox_center(ns.bbox)) <= max_center_distance_mm:
            return True
    return False


def _rule_bedside_lamp_not_independent(
    bed: StyledObject | None,
    nightstands: list[StyledObject],
    bedside_lamps: list[StyledObject],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if bed is None:
        return RuleResult(
            rule_id="bedside_lamp/not_independent",
            name="Bedside lamp near bed + nightstand",
            weight=weight,
            earned=0,
            status="failed",
            details="No bed found.",
        )
    if not bedside_lamps:
        return RuleResult(
            rule_id="bedside_lamp/not_independent",
            name="Bedside lamp near bed + nightstand",
            weight=weight,
            earned=0,
            status="failed",
            details="No bedside_lamp found.",
        )

    bed_center = _bbox_center(bed.bbox)
    ns_centers = [_bbox_center(ns.bbox) for ns in nightstands]
    ok = False
    for lamp in bedside_lamps:
        lamp_center = _bbox_center(lamp.bbox)
        d_bed = _euclidean(lamp_center, bed_center)
        if d_bed > cfg.bedside_lamp_near_bed_mm:
            continue
        if not ns_centers:
            ok = True
            break
        if (
            min(_euclidean(lamp_center, c) for c in ns_centers)
            <= cfg.bedside_lamp_near_nightstand_mm
        ):
            ok = True
            break

    return RuleResult(
        rule_id="bedside_lamp/not_independent",
        name="Bedside lamp near bed + nightstand",
        weight=weight,
        earned=(weight if ok else 0),
        status=("passed" if ok else "failed"),
        details=(
            f"near_bed_mm<={cfg.bedside_lamp_near_bed_mm} and near_nightstand_mm<={cfg.bedside_lamp_near_nightstand_mm}."
        ),
    )


def _rule_wardrobe_against_wall(
    wardrobe: StyledObject | None,
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 9
    if wardrobe is None:
        return RuleResult(
            rule_id="wardrobe/against_wall",
            name="Wardrobe near a wall",
            weight=weight,
            earned=0,
            status="failed",
            details="No wardrobe found (object_type=wardrobe or closet).",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="wardrobe/against_wall",
            name="Wardrobe near a wall",
            weight=weight,
            earned=0,
            status="skipped",
            details="Room polygon missing/invalid.",
        )

    d = _min_distance_bbox_to_polygon(wardrobe.bbox, room_poly)
    passed = d <= cfg.object_against_wall_mm
    return RuleResult(
        rule_id="wardrobe/against_wall",
        name="Wardrobe near a wall",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"wardrobe={wardrobe.instance_id}; min_distance_to_wall_mm={round(d, 1)}.",
    )


def _rule_wardrobe_near_door_than_bed(
    wardrobe: StyledObject | None,
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    door_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> RuleResult:
    weight = 7
    if wardrobe is None:
        return RuleResult(
            rule_id="wardrobe/near_door",
            name="Wardrobe closer to door than bed headboard",
            weight=weight,
            earned=0,
            status="failed",
            details="No wardrobe found.",
        )
    if bed is None or headboard_edge is None:
        return RuleResult(
            rule_id="wardrobe/near_door",
            name="Wardrobe closer to door than bed headboard",
            weight=weight,
            earned=0,
            status="failed",
            details="Missing bed/headboard edge.",
        )
    if not door_segments:
        return RuleResult(
            rule_id="wardrobe/near_door",
            name="Wardrobe closer to door than bed headboard",
            weight=weight,
            earned=0,
            status="failed",
            details="No door opening segment found.",
        )

    door_mid = _segment_midpoint(door_segments[0][1], door_segments[0][2])
    w_d = _euclidean(_bbox_center(wardrobe.bbox), door_mid)
    b_d = _euclidean(headboard_edge.midpoint, door_mid)
    passed = w_d <= b_d
    return RuleResult(
        rule_id="wardrobe/near_door",
        name="Wardrobe closer to door than bed headboard",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"dist_wardrobe_to_door_mm={round(w_d, 1)}; dist_headboard_to_door_mm={round(b_d, 1)}.",
    )


def _rule_wardrobe_not_next_to_headboard(
    wardrobe: StyledObject | None,
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if wardrobe is None:
        return RuleResult(
            rule_id="wardrobe/not_next_to_headboard",
            name="Wardrobe not next to headboard area",
            weight=weight,
            earned=0,
            status="failed",
            details="No wardrobe found.",
        )
    if bed is None or headboard_edge is None:
        return RuleResult(
            rule_id="wardrobe/not_next_to_headboard",
            name="Wardrobe not next to headboard area",
            weight=weight,
            earned=0,
            status="failed",
            details="Missing bed/headboard edge.",
        )

    d = _min_distance_bbox_to_segment(wardrobe.bbox, headboard_edge.a, headboard_edge.b)
    passed = d >= cfg.wardrobe_far_from_headboard_mm
    return RuleResult(
        rule_id="wardrobe/not_next_to_headboard",
        name="Wardrobe not next to headboard area",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"min_distance_wardrobe_to_headboard_mm={round(d, 1)} (min={cfg.wardrobe_far_from_headboard_mm}).",
    )


def _rule_wardrobe_front_clearance(
    wardrobe: StyledObject | None,
    objects: list[StyledObject],
    room_poly: list[Point],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 10
    if wardrobe is None:
        return RuleResult(
            rule_id="wardrobe/front_clearance",
            name="Wardrobe has front clearance",
            weight=weight,
            earned=0,
            status="failed",
            details="No wardrobe found.",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="wardrobe/front_clearance",
            name="Wardrobe has front clearance",
            weight=weight,
            earned=0,
            status="failed",
            details="Room polygon missing/invalid.",
        )

    clearance = _wardrobe_front_clearance_rect(
        wardrobe.bbox, room_poly, cfg.wardrobe_front_clearance_mm
    )
    blockers = [
        o for o in objects if _is_floor_blocker(o, exclude_ids={wardrobe.instance_id})
    ]
    hits = [o.instance_id for o in blockers if _rects_intersect(clearance, o.bbox)]
    passed = not hits
    return RuleResult(
        rule_id="wardrobe/front_clearance",
        name="Wardrobe has front clearance",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"clearance_depth_mm={cfg.wardrobe_front_clearance_mm}; blockers_in_clearance={hits}.",
    )


def _rule_side_table_near_window(
    side_table: StyledObject | None,
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 7
    if side_table is None:
        return RuleResult(
            rule_id="side_table/near_window",
            name="Side table near window (if present)",
            weight=weight,
            earned=0,
            status="failed",
            details="No side_table found.",
        )
    if not window_segments:
        return RuleResult(
            rule_id="side_table/near_window",
            name="Side table near window (if present)",
            weight=weight,
            earned=0,
            status="failed",
            details="No window opening segment found.",
        )

    center = _bbox_center(side_table.bbox)
    d = min(_distance_point_to_segment(center, a, b) for _, a, b in window_segments)
    passed = d <= cfg.side_table_near_window_mm
    return RuleResult(
        rule_id="side_table/near_window",
        name="Side table near window (if present)",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=(
            f"side_table={side_table.instance_id}; "
            f"min_distance_to_window_mm={round(d, 1)} (threshold={cfg.side_table_near_window_mm})."
        ),
    )


def _rule_side_table_alternative_placement(
    side_table: StyledObject | None,
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    room_poly: list[Point],
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 6
    if side_table is None:
        return RuleResult(
            rule_id="side_table/alternative_placement",
            name="Side table alternative placement when no window",
            weight=weight,
            earned=0,
            status="failed",
            details="No side_table found.",
        )
    if window_segments:
        return RuleResult(
            rule_id="side_table/alternative_placement",
            name="Side table alternative placement when no window",
            weight=weight,
            earned=0,
            status="failed",
            details="Window exists; fallback placement rule therefore fails by rubric standardization.",
        )
    if bed is None or headboard_edge is None or len(room_poly) < 3:
        return RuleResult(
            rule_id="side_table/alternative_placement",
            name="Side table alternative placement when no window",
            weight=weight,
            earned=0,
            status="failed",
            details="Missing bed/headboard/room polygon.",
        )

    side_table_center = _bbox_center(side_table.bbox)
    bed_distance = _distance_point_to_rect(side_table_center, bed.bbox)
    near_bed = bed_distance <= cfg.side_table_near_bed_mm

    room_bbox = _polygon_bbox(room_poly)
    opposite_wall_distance = _distance_point_to_opposite_wall(
        side_table_center, room_bbox, headboard_edge
    )
    near_opposite_wall = (
        opposite_wall_distance is not None
        and opposite_wall_distance <= cfg.side_table_opposite_wall_mm
    )

    passed = bool(near_bed or near_opposite_wall)
    return RuleResult(
        rule_id="side_table/alternative_placement",
        name="Side table alternative placement when no window",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=(
            f"near_bed={near_bed}; near_opposite_wall={near_opposite_wall}; "
            f"bed_distance_mm={round(bed_distance, 1)}; "
            f"opposite_wall_distance_mm={round(opposite_wall_distance, 1) if opposite_wall_distance is not None else None}."
        ),
    )


def _rule_side_table_not_blocking_paths(
    side_table: StyledObject | None,
    bed: StyledObject | None,
    wardrobe: StyledObject | None,
    door_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> RuleResult:
    weight = 9
    if side_table is None:
        return RuleResult(
            rule_id="side_table/not_block_path",
            name="Side table does not block key paths",
            weight=weight,
            earned=0,
            status="failed",
            details="No side_table found.",
        )
    if bed is None or not door_segments:
        return RuleResult(
            rule_id="side_table/not_block_path",
            name="Side table does not block key paths",
            weight=weight,
            earned=0,
            status="failed",
            details="Missing bed or door segment.",
        )

    door_mid = _segment_midpoint(door_segments[0][1], door_segments[0][2])
    bed_mid = _bbox_center(bed.bbox)
    hits: list[str] = []

    if _segment_intersects_rect(door_mid, bed_mid, side_table.bbox):
        hits.append("door_to_bed")
    if wardrobe is not None:
        if _segment_intersects_rect(
            bed_mid, _bbox_center(wardrobe.bbox), side_table.bbox
        ):
            hits.append("bed_to_wardrobe")

    passed = not hits
    return RuleResult(
        rule_id="side_table/not_block_path",
        name="Side table does not block key paths",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"intersections={hits}.",
    )


def _rule_curtain_present_with_window(
    curtains: list[StyledObject],
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> RuleResult:
    weight = 10
    if not window_segments:
        return RuleResult(
            rule_id="curtain/present_with_window",
            name="Curtain present when window exists",
            weight=weight,
            earned=0,
            status="failed",
            details="No window opening segment found.",
        )
    if not curtains:
        return RuleResult(
            rule_id="curtain/present_with_window",
            name="Curtain present when window exists",
            weight=weight,
            earned=0,
            status="failed",
            details="No curtain found.",
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
        rule_id="curtain/present_with_window",
        name="Curtain present when window exists",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"curtains={len(curtains)}; curtains_attached_to_window={len(attached)}.",
    )


def _rule_bed_not_under_window(
    bed: StyledObject | None,
    headboard_edge: _RectEdge | None,
    window_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if bed is None or headboard_edge is None:
        return RuleResult(
            rule_id="bed/avoid_window",
            name="Bed headboard not under a window",
            weight=weight,
            earned=0,
            status="failed",
            details="No bed found (or cannot infer headboard edge).",
        )
    if not window_segments:
        return RuleResult(
            rule_id="bed/avoid_window",
            name="Bed headboard not under a window",
            weight=weight,
            earned=0,
            status="failed",
            details="No window opening segment found.",
        )

    d = min(
        _distance_point_to_segment(headboard_edge.midpoint, a, b)
        for _, a, b in window_segments
    )
    passed = d >= cfg.bed_avoid_window_headboard_mm
    return RuleResult(
        rule_id="bed/avoid_window",
        name="Bed headboard not under a window",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"min_headboard_to_window_mm={round(d, 1)} (min={cfg.bed_avoid_window_headboard_mm}).",
    )


def _rule_ceiling_light_near_center(
    ceiling_lights: list[StyledObject],
    room_poly: list[Point],
) -> RuleResult:
    weight = 9
    if not ceiling_lights:
        return RuleResult(
            rule_id="ceiling_light/near_center",
            name="Ceiling light near room center",
            weight=weight,
            earned=0,
            status="failed",
            details="No ceiling_light found.",
        )
    if len(room_poly) < 3:
        return RuleResult(
            rule_id="ceiling_light/near_center",
            name="Ceiling light near room center",
            weight=weight,
            earned=0,
            status="failed",
            details="Room polygon missing/invalid.",
        )

    room_center = _polygon_centroid(room_poly)
    room_bbox = _polygon_bbox(room_poly)
    diag = math.hypot(
        room_bbox.max_x - room_bbox.min_x, room_bbox.max_y - room_bbox.min_y
    )
    if diag <= 0:
        diag = 1.0

    best = min(
        ceiling_lights, key=lambda l: _euclidean(_bbox_center(l.bbox), room_center)
    )
    d = _euclidean(_bbox_center(best.bbox), room_center)
    passed = d <= 0.35 * diag
    return RuleResult(
        rule_id="ceiling_light/near_center",
        name="Ceiling light near room center",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"closest_ceiling_light={best.instance_id}; dist_to_center_mm={round(d, 1)}; diag_mm={round(diag, 1)}.",
    )


def _rule_ceiling_light_hung_on_ceiling(
    ceiling_lights: list[StyledObject],
) -> RuleResult:
    weight = 8
    if not ceiling_lights:
        return RuleResult(
            rule_id="ceiling_light/hung_on_ceiling",
            name="Ceiling light hung on ceiling",
            weight=weight,
            earned=0,
            status="failed",
            details="No ceiling_light found.",
        )

    ok = [
        l
        for l in ceiling_lights
        if l.place_on is not None
        and l.place_on.method == "hang_on"
        and l.place_on.target_instance_id == "__ceiling__"
    ]
    passed = bool(ok)
    return RuleResult(
        rule_id="ceiling_light/hung_on_ceiling",
        name="Ceiling light hung on ceiling",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"hung_on_ceiling={len(ok)}/{len(ceiling_lights)}.",
    )


def _rule_ceiling_light_not_tied_to_single_object(
    ceiling_lights: list[StyledObject],
    anchors: list[StyledObject | None],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if not ceiling_lights:
        return RuleResult(
            rule_id="ceiling_light/not_tied_to_object",
            name="Ceiling light not tied to single object",
            weight=weight,
            earned=0,
            status="failed",
            details="No ceiling_light found.",
        )

    anchor_boxes = [a.bbox for a in anchors if a is not None]
    if not anchor_boxes:
        return RuleResult(
            rule_id="ceiling_light/not_tied_to_object",
            name="Ceiling light not tied to single object",
            weight=weight,
            earned=0,
            status="skipped",
            details="No anchor objects found.",
        )

    not_over_anchor = []
    for light in ceiling_lights:
        center = _bbox_center(light.bbox)
        if not any(
            _point_in_bbox(
                center, bbox, padding_mm=cfg.ceiling_light_near_object_max_mm
            )
            for bbox in anchor_boxes
        ):
            not_over_anchor.append(light.instance_id)

    passed = bool(not_over_anchor)
    return RuleResult(
        rule_id="ceiling_light/not_tied_to_object",
        name="Ceiling light not tied to single object",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=f"not_over_anchor={len(not_over_anchor)}/{len(ceiling_lights)}.",
    )


def _rule_ceiling_light_general_overhead(
    ceiling_lights: list[StyledObject],
    anchors: list[StyledObject | None],
    cfg: ScoringConfig,
) -> RuleResult:
    weight = 8
    if not ceiling_lights:
        return RuleResult(
            rule_id="ceiling_light/general_overhead",
            name="Ceiling light is general overhead lighting",
            weight=weight,
            earned=0,
            status="failed",
            details="No ceiling_light found.",
        )

    anchor_boxes = [a.bbox for a in anchors if a is not None]
    valid_ids: list[str] = []
    reasons: list[str] = []
    for light in ceiling_lights:
        hung_on_ceiling = (
            light.place_on is not None
            and light.place_on.method == "hang_on"
            and light.place_on.target_instance_id == "__ceiling__"
        )
        center = _bbox_center(light.bbox)
        tied_to_single_object = (
            any(
                _point_in_bbox(
                    center, bbox, padding_mm=cfg.ceiling_light_near_object_max_mm
                )
                for bbox in anchor_boxes
            )
            if anchor_boxes
            else False
        )
        if hung_on_ceiling and not tied_to_single_object:
            valid_ids.append(light.instance_id)
        else:
            reasons.append(
                f"{light.instance_id}: hung_on_ceiling={hung_on_ceiling}, tied_to_single_object={tied_to_single_object}"
            )

    passed = bool(valid_ids)
    details = (
        f"valid_general_lights={valid_ids}."
        if passed
        else "No ceiling light satisfied both conditions: hung on ceiling and not tied to a single object. "
        + "; ".join(reasons)
    )
    return RuleResult(
        rule_id="ceiling_light/general_overhead",
        name="Ceiling light is general overhead lighting",
        weight=weight,
        earned=(weight if passed else 0),
        status=("passed" if passed else "failed"),
        details=details,
    )


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


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.min_x + bbox.max_x) / 2.0, (bbox.min_y + bbox.max_y) / 2.0)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_in_bbox(
    p: tuple[float, float],
    bbox: BBox,
    *,
    padding_mm: int = 0,
) -> bool:
    x, y = p
    return (
        bbox.min_x - padding_mm <= x <= bbox.max_x + padding_mm
        and bbox.min_y - padding_mm <= y <= bbox.max_y + padding_mm
    )


def _distance_point_to_rect(p: tuple[float, float], rect: BBox) -> float:
    x, y = p
    dx = max(rect.min_x - x, 0.0, x - rect.max_x)
    dy = max(rect.min_y - y, 0.0, y - rect.max_y)
    return math.hypot(dx, dy)


def _segment_midpoint(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


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
    if len(poly) < 2:
        return []
    pts = [(float(p.x), float(p.y)) for p in poly]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _min_distance_point_to_polygon_edges(
    p: tuple[float, float], poly: list[Point]
) -> float:
    return min(_distance_point_to_segment(p, a, b) for a, b in _polygon_edges(poly))


def _min_distance_bbox_to_polygon(bbox: BBox, poly: list[Point]) -> float:
    points = _bbox_sample_points(bbox)
    return min(_min_distance_point_to_polygon_edges(p, poly) for p in points)


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
    ]


def _min_distance_bbox_to_segment(
    bbox: BBox, a: tuple[float, float], b: tuple[float, float]
) -> float:
    points = _bbox_sample_points(bbox)
    return min(_distance_point_to_segment(p, a, b) for p in points)


def _is_rect_adjacent_to_bed_edge(
    rect: BBox,
    bed: BBox,
    edge: _RectEdge,
    cfg: ScoringConfig,
) -> bool:
    if edge.side in {"top", "bottom"}:
        overlap = _interval_overlap(rect.min_x, rect.max_x, bed.min_x, bed.max_x)
        if overlap <= 0:
            return False
        if edge.side == "top":
            gap = bed.min_y - rect.max_y
        else:
            gap = rect.min_y - bed.max_y
        return (-cfg.nightstand_overlap_tol_mm) <= gap <= cfg.nightstand_adjacent_gap_mm

    overlap = _interval_overlap(rect.min_y, rect.max_y, bed.min_y, bed.max_y)
    if overlap <= 0:
        return False
    if edge.side == "left":
        gap = bed.min_x - rect.max_x
    else:
        gap = rect.min_x - bed.max_x
    return (-cfg.nightstand_overlap_tol_mm) <= gap <= cfg.nightstand_adjacent_gap_mm


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def _wardrobe_front_clearance_rect(
    wardrobe: BBox, room_poly: list[Point], depth_mm: int
) -> BBox:
    edges = _rect_edges(wardrobe)
    back = min(
        edges, key=lambda e: _min_distance_point_to_polygon_edges(e.midpoint, room_poly)
    )
    front_side = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[back.side]
    min_x, min_y, max_x, max_y = (
        wardrobe.min_x,
        wardrobe.min_y,
        wardrobe.max_x,
        wardrobe.max_y,
    )
    if front_side == "right":
        return BBox(min_x=max_x, min_y=min_y, max_x=max_x + depth_mm, max_y=max_y)
    if front_side == "left":
        return BBox(min_x=min_x - depth_mm, min_y=min_y, max_x=min_x, max_y=max_y)
    if front_side == "bottom":
        return BBox(min_x=min_x, min_y=max_y, max_x=max_x, max_y=max_y + depth_mm)
    return BBox(min_x=min_x, min_y=min_y - depth_mm, max_x=max_x, max_y=min_y)


def _is_floor_blocker(obj: StyledObject, *, exclude_ids: set[str]) -> bool:
    if obj.instance_id in exclude_ids:
        return False
    if obj.place_on is None:
        return True
    return obj.place_on.method == "floor"


def _rects_intersect(a: BBox, b: BBox) -> bool:
    return not (
        a.max_x <= b.min_x
        or a.min_x >= b.max_x
        or a.max_y <= b.min_y
        or a.min_y >= b.max_y
    )


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
    if _point_in_rect(a, r) or _point_in_rect(b, r):
        return True
    edges = [
        ((r.min_x, r.min_y), (r.max_x, r.min_y)),
        ((r.max_x, r.min_y), (r.max_x, r.max_y)),
        ((r.max_x, r.max_y), (r.min_x, r.max_y)),
        ((r.min_x, r.max_y), (r.min_x, r.min_y)),
    ]
    return any(_segments_intersect(a, b, e1, e2) for e1, e2 in edges)


def _point_in_rect(p: tuple[float, float], r: BBox) -> bool:
    return r.min_x <= p[0] <= r.max_x and r.min_y <= p[1] <= r.max_y


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
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    a *= 0.5
    cx /= 6.0 * a
    cy /= 6.0 * a
    return (cx, cy)


def _polygon_bbox(poly: list[Point]) -> BBox:
    xs = [p.x for p in poly]
    ys = [p.y for p in poly]
    return BBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _distance_point_to_opposite_wall(
    p: tuple[float, float],
    room_bbox: BBox,
    headboard_edge: _RectEdge,
) -> float | None:
    x, y = p
    if headboard_edge.side == "top":
        return abs(y - room_bbox.min_y)
    if headboard_edge.side == "bottom":
        return abs(y - room_bbox.max_y)
    if headboard_edge.side == "left":
        return abs(x - room_bbox.max_x)
    if headboard_edge.side == "right":
        return abs(x - room_bbox.min_x)
    return None


def _bedroom_key_objects(
    objects_by_cat: dict[str, list[StyledObject]],
) -> list[StyledObject]:
    out: list[StyledObject] = []
    for cat in ("bed", "nightstand", "wardrobe", "side_table"):
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
    samples = _bbox_sample_points(bbox)
    samples.append(_bbox_center(bbox))
    return all(_point_in_polygon_or_on_edge(p, room_poly) for p in samples)


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
