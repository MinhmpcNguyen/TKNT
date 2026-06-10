from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evaluation.layout_benchmark.normalize import load_output_payload, normalize_payload
from evaluation.layout_benchmark.schema import (
    Point2D,
    normalized_layout_to_json,
    points_from_json,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = BENCHMARK_DIR / "runs" / "manual" / "runs"
DEFAULT_CASES_PATH = BENCHMARK_DIR / "cases.json"

# Change these names when the run folders change.
SYSTEM_RUN_FOLDERS = ("tknt_new_s_o", "idesign_s_o", "flairgpt_s_o")

DEFAULT_OUTPUT_NAME = "layout_2d.png"
GRID_SIZE_MM = 500.0
ROOM_FILL = "#faf7f1"
ROOM_STROKE = "#b8b0a2"
GRID_STROKE = "#e6e1d8"
WALL_STROKE = "#2f2f2f"
WINDOW_STROKE = "#4f9fbd"
WINDOW_LABEL_FILL = "#edf8ff"
WINDOW_LABEL_STROKE = "#8ec6d9"
WINDOW_LABEL_TEXT = "#2e6f86"
DOOR_STROKE = "#2f3437"
DOOR_LABEL_FILL = "#fff3e7"
DOOR_LABEL_STROKE = "#d2a06a"
DOOR_LABEL_TEXT = "#704417"

OpeningKind = Literal["door", "window"]

_LOGGER = logging.getLogger(__name__)

_TYPE_COLORS: dict[str, str] = {
    "armchair": "#9eb7a8",
    "bed": "#d9c8b8",
    "bookshelf": "#b98a61",
    "ceiling_lamp": "#f0d47a",
    "chair": "#8fa0a3",
    "coffee_table": "#d0a36f",
    "curtain": "#c9866a",
    "desk": "#b78b72",
    "dresser": "#a57c67",
    "floor_lamp": "#dfc96f",
    "nightstand": "#7b5a43",
    "rug": "#8fa09a",
    "sofa": "#9eb7a8",
    "table_lamp": "#e9cd76",
    "tv": "#3f4448",
    "tv_console": "#938679",
    "wardrobe": "#c9b39a",
}

_SOFT_LAYER_TOKENS = (
    "ceiling_lamp",
    "ceiling_light",
    "curtain",
    "lamp",
    "light",
    "mirror",
    "painting",
    "pendant",
    "wall_art",
)

_FLOOR_LAYER_TOKENS = ("carpet", "rug")

_DIRECTION_SIDE_VECTORS: dict[str, Point2D] = {
    "bottom": Point2D(x=0.0, y=-1.0),
    "down": Point2D(x=0.0, y=-1.0),
    "east": Point2D(x=1.0, y=0.0),
    "east_wall": Point2D(x=1.0, y=0.0),
    "left": Point2D(x=-1.0, y=0.0),
    "north": Point2D(x=0.0, y=1.0),
    "north_wall": Point2D(x=0.0, y=1.0),
    "right": Point2D(x=1.0, y=0.0),
    "south": Point2D(x=0.0, y=-1.0),
    "south_wall": Point2D(x=0.0, y=-1.0),
    "top": Point2D(x=0.0, y=1.0),
    "up": Point2D(x=0.0, y=1.0),
    "west": Point2D(x=-1.0, y=0.0),
    "west_wall": Point2D(x=-1.0, y=0.0),
}

_ORIENTATION_KEYS = (
    "axis_world",
    "front_side_world",
    "front_world",
    "rotation_ccw",
)


@dataclass(frozen=True)
class OpeningView:
    opening_id: str
    kind: OpeningKind
    start: Point2D
    end: Point2D
    clearance_mm: float


@dataclass(frozen=True)
class ObjectView:
    object_id: str
    object_type: str
    polygon: tuple[Point2D, ...]
    rotation_degrees: float
    raw: Mapping[str, object]


@dataclass(frozen=True)
class LayoutView:
    case_id: str
    system_id: str
    room_polygon: tuple[Point2D, ...]
    openings: tuple[OpeningView, ...]
    objects: tuple[ObjectView, ...]
    source_path: Path
    status_note: str | None = None


@dataclass(frozen=True)
class VisualizeResult:
    case_dir: Path
    output_path: Path | None
    status: Literal["written", "skipped", "failed"]
    message: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render normalized layout benchmark runs as UI-like 2D PNGs."
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Root containing run folders such as tknt_new, idesign, and flairgpt.",
    )
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEM_RUN_FOLDERS),
        help="Comma-separated run folder names under --runs-root.",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="Benchmark cases JSON used as a fallback when normalized.json is missing.",
    )
    parser.add_argument(
        "--include-cases",
        default="",
        help=(
            "Optional comma-separated case IDs or case folder names. "
            "Empty means every case folder under each selected system."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="PNG filename written inside every case folder.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite an existing PNG.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any selected case fails to render.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-case progress.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    results = visualize_runs(
        runs_root=Path(args.runs_root),
        system_folders=_split_csv(args.systems),
        cases_path=Path(args.cases),
        include_cases=frozenset(_split_csv(args.include_cases)),
        output_name=args.output_name,
        dpi=max(72, args.dpi),
        skip_existing=args.skip_existing,
    )

    written = sum(result.status == "written" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    _LOGGER.info(
        "Visualization complete: %s written, %s skipped, %s failed.",
        written,
        skipped,
        failed,
    )
    return 1 if args.strict and failed else 0


def visualize_runs(
    *,
    runs_root: Path,
    system_folders: Sequence[str],
    cases_path: Path,
    include_cases: frozenset[str],
    output_name: str,
    dpi: int,
    skip_existing: bool,
) -> list[VisualizeResult]:
    case_index = _load_case_index(cases_path)
    results: list[VisualizeResult] = []
    for system_folder in system_folders:
        system_dir = runs_root / system_folder
        if not system_dir.is_dir():
            message = f"System folder not found: {system_dir}"
            _LOGGER.warning(message)
            results.append(
                VisualizeResult(
                    case_dir=system_dir,
                    output_path=None,
                    status="failed",
                    message=message,
                )
            )
            continue

        for case_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
            case_id = _case_id_from_dir_name(case_dir.name)
            if (
                include_cases
                and case_id not in include_cases
                and case_dir.name not in include_cases
            ):
                continue
            result = visualize_case_dir(
                case_dir=case_dir,
                system_folder=system_folder,
                case_index=case_index,
                output_name=output_name,
                dpi=dpi,
                skip_existing=skip_existing,
            )
            results.append(result)
            if result.status == "failed":
                _LOGGER.warning("%s: %s", case_dir, result.message)
            elif result.status == "written":
                _LOGGER.info("Wrote %s", result.output_path)
            else:
                _LOGGER.debug("%s: %s", case_dir, result.message)
    return results


def visualize_case_dir(
    *,
    case_dir: Path,
    system_folder: str,
    case_index: Mapping[str, Mapping[str, object]],
    output_name: str,
    dpi: int,
    skip_existing: bool,
) -> VisualizeResult:
    output_path = case_dir / output_name
    if skip_existing and output_path.exists():
        return VisualizeResult(
            case_dir=case_dir,
            output_path=output_path,
            status="skipped",
            message="PNG already exists.",
        )

    try:
        layout = load_layout_for_case_dir(
            case_dir=case_dir,
            system_folder=system_folder,
            case_index=case_index,
        )
        render_layout_png(layout=layout, output_path=output_path, dpi=dpi)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return VisualizeResult(
            case_dir=case_dir,
            output_path=None,
            status="failed",
            message=str(exc),
        )
    return VisualizeResult(
        case_dir=case_dir,
        output_path=output_path,
        status="written",
        message="Rendered layout PNG.",
    )


def load_layout_for_case_dir(
    *,
    case_dir: Path,
    system_folder: str,
    case_index: Mapping[str, Mapping[str, object]],
) -> LayoutView:
    normalized_path = case_dir / "normalized.json"
    if normalized_path.is_file():
        payload = _load_json_mapping(normalized_path)
        payload = _enrich_normalized_orientations(payload=payload, case_dir=case_dir)
        return _layout_view_from_payload(
            payload=payload,
            source_path=normalized_path,
            status_note=None,
        )

    fallback_payload, source_path, status_note = _normalize_case_output(
        case_dir=case_dir,
        system_folder=system_folder,
        case_index=case_index,
    )
    fallback_payload = _enrich_normalized_orientations(
        payload=fallback_payload,
        case_dir=case_dir,
    )
    return _layout_view_from_payload(
        payload=fallback_payload,
        source_path=source_path,
        status_note=status_note,
    )


def _enrich_normalized_orientations(
    *,
    payload: Mapping[str, object],
    case_dir: Path,
) -> Mapping[str, object]:
    orientation_rows = _load_canonical_orientation_rows(case_dir)
    orientation_by_id: dict[str, Mapping[str, object]] = {}
    for row in orientation_rows:
        object_id = _first_string(row, "object_id", "instance_id", "id")
        orientation = {key: row[key] for key in _ORIENTATION_KEYS if key in row}
        if object_id and orientation:
            orientation_by_id[object_id] = orientation
    if not orientation_by_id:
        return payload

    changed = False
    objects: list[object] = []
    for value in _sequence(payload.get("objects")):
        item = _mapping(value)
        object_id = _first_string(item, "id", "object_id", "instance_id")
        orientation = orientation_by_id.get(object_id)
        if not item or orientation is None:
            objects.append(value)
            continue

        enriched_item = dict(item)
        raw = dict(_mapping(enriched_item.get("raw")))
        for key, orientation_value in orientation.items():
            if key not in raw:
                raw[key] = orientation_value
                changed = True
        rotation_degrees = _number(orientation.get("rotation_ccw"))
        if rotation_degrees is not None:
            enriched_item["rotation_degrees"] = rotation_degrees
            changed = True
        enriched_item["raw"] = raw
        objects.append(enriched_item)

    if not changed:
        return payload
    enriched_payload = dict(payload)
    enriched_payload["objects"] = objects
    return enriched_payload


def _load_canonical_orientation_rows(
    case_dir: Path,
) -> tuple[Mapping[str, object], ...]:
    paths = (
        case_dir / "module_outputs" / "layout_variants.json",
        case_dir / "09_layout_variants.json",
        case_dir / "module_outputs" / "absolute_layout.json",
        case_dir / "07_absolute_layout.json",
    )
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = _load_json_mapping(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _LOGGER.debug("Cannot load orientation metadata from %s: %s", path, exc)
            continue
        absolute_layout = _canonical_absolute_layout(payload)
        rows = tuple(
            row
            for value in _sequence(absolute_layout.get("objects"))
            if (row := _mapping(value)) and any(key in row for key in _ORIENTATION_KEYS)
        )
        if rows:
            return rows
    return ()


def _canonical_absolute_layout(payload: Mapping[str, object]) -> Mapping[str, object]:
    variants = _sequence(payload.get("variants"))
    if variants:
        selected_variant_id = _first_string(payload, "selected_variant_id")
        selected_variant = next(
            (
                variant
                for value in variants
                if (variant := _mapping(value))
                and _first_string(variant, "variant_id") == selected_variant_id
            ),
            _mapping(variants[0]),
        )
        return _mapping(selected_variant.get("absolute_layout"))
    return _mapping(payload.get("absolute_layout")) or payload


def render_layout_png(*, layout: LayoutView, output_path: Path, dpi: int) -> None:
    _configure_matplotlib()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MatplotlibPolygon
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required. Try running with "
            ".conda_envs/benchmark-flairgpt/bin/python."
        ) from exc

    extents = _layout_extents(layout)
    width_in, height_in = _figure_size(extents)
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")

    _draw_grid(ax=ax, extents=extents)
    ax.add_patch(
        MatplotlibPolygon(
            _xy_rows(layout.room_polygon),
            closed=True,
            facecolor=ROOM_FILL,
            edgecolor=ROOM_STROKE,
            linewidth=2.2,
            joinstyle="round",
            zorder=2,
        )
    )
    _draw_openings(
        ax=ax,
        openings=layout.openings,
        room_polygon=layout.room_polygon,
    )
    for row in sorted(layout.objects, key=_object_layer):
        color = _object_color(row)
        alpha = 0.52 if _has_any_token(row.object_type, _FLOOR_LAYER_TOKENS) else 0.92
        edge_color = "#4b4843" if alpha > 0.6 else "#6d766e"
        line_width = 1.35 if alpha > 0.6 else 1.0
        ax.add_patch(
            MatplotlibPolygon(
                _xy_rows(row.polygon),
                closed=True,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=alpha,
                joinstyle="round",
                zorder=4 + _object_layer(row),
            )
        )
        _draw_front_marker(ax=ax, row=row)
        _draw_object_label(ax=ax, row=row)

    if not layout.objects:
        center = _polygon_center(layout.room_polygon)
        ax.text(
            center.x,
            center.y,
            "No placed objects",
            ha="center",
            va="center",
            fontsize=9,
            color="#a15a14",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#fff1e3",
                "edgecolor": "#e7c7a5",
                "alpha": 0.92,
            },
            zorder=20,
        )

    _apply_axes_layout(ax=ax, extents=extents)
    title = f"{layout.case_id} · {layout.system_id}"
    if layout.status_note:
        title = f"{title} · {layout.status_note}"
    fig.suptitle(title, fontsize=10, color="#2f2f2f", y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _normalize_case_output(
    *,
    case_dir: Path,
    system_folder: str,
    case_index: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, object], Path, str | None]:
    output_path = case_dir / "output.json"
    if not output_path.is_file():
        raise ValueError(f"Missing both normalized.json and output.json in {case_dir}")

    payload = load_output_payload(output_path)
    payload_mapping = _mapping(payload)
    case_id = _case_id_from_dir_name(case_dir.name)
    benchmark_case = case_index.get(case_id) or _case_from_input(case_dir, case_id)
    normalizer = _infer_normalizer(system_folder=system_folder, case_dir=case_dir)
    system_id = _first_string(payload_mapping, "system_id") or _system_id_from_dir_name(
        case_dir.name,
        system_folder,
    )
    layout = normalize_payload(
        normalizer=normalizer,
        payload=payload,
        benchmark_case=benchmark_case,
        system_id=system_id,
        source_path=output_path,
    )
    status_note = _first_string(payload_mapping, "error") or None
    return normalized_layout_to_json(layout), output_path, status_note


def _layout_view_from_payload(
    *,
    payload: Mapping[str, object],
    source_path: Path,
    status_note: str | None,
) -> LayoutView:
    room_polygon = points_from_json(_sequence(payload.get("room_polygon")))
    if len(room_polygon) < 3:
        raise ValueError(f"Layout has no valid room polygon: {source_path}")

    openings: list[OpeningView] = []
    for row in _sequence(payload.get("openings")):
        item = _mapping(row)
        kind = _opening_kind(item.get("kind"))
        if kind is None:
            continue
        start = _point(item.get("start"))
        end = _point(item.get("end"))
        if start is None or end is None:
            continue
        openings.append(
            OpeningView(
                opening_id=_first_string(item, "id", "opening_id") or kind,
                kind=kind,
                start=start,
                end=end,
                clearance_mm=_number(item.get("clearance_mm")) or 0.0,
            )
        )

    objects: list[ObjectView] = []
    for row in _sequence(payload.get("objects")):
        item = _mapping(row)
        polygon = points_from_json(_sequence(item.get("polygon")))
        if len(polygon) < 3:
            continue
        raw = dict(_mapping(item.get("raw")))
        for key in _ORIENTATION_KEYS:
            if key in item and key not in raw:
                raw[key] = item[key]
        if "rotation_degrees" in item and "rotation_degrees" not in raw:
            raw["rotation_degrees"] = item["rotation_degrees"]
        objects.append(
            ObjectView(
                object_id=_first_string(item, "id", "object_id", "instance_id")
                or f"object_{len(objects) + 1}",
                object_type=_first_string(item, "type", "object_type", "category")
                or "object",
                polygon=polygon,
                rotation_degrees=_number(item.get("rotation_degrees")) or 0.0,
                raw=raw,
            )
        )

    return LayoutView(
        case_id=_first_string(payload, "case_id")
        or _case_id_from_dir_name(source_path.parent.name),
        system_id=_first_string(payload, "system_id") or source_path.parent.name,
        room_polygon=room_polygon,
        openings=tuple(openings),
        objects=tuple(objects),
        source_path=source_path,
        status_note=status_note,
    )


def _configure_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "tknt_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")


def _draw_grid(*, ax: object, extents: tuple[float, float, float, float]) -> None:
    min_x, min_y, max_x, max_y = extents
    start_x = _floor_to_grid(min_x)
    end_x = _ceil_to_grid(max_x)
    start_y = _floor_to_grid(min_y)
    end_y = _ceil_to_grid(max_y)
    x = start_x
    while x <= end_x:
        ax.plot([x, x], [start_y, end_y], color=GRID_STROKE, linewidth=0.45, zorder=0)
        x += GRID_SIZE_MM
    y = start_y
    while y <= end_y:
        ax.plot([start_x, end_x], [y, y], color=GRID_STROKE, linewidth=0.45, zorder=0)
        y += GRID_SIZE_MM


def _draw_openings(
    *,
    ax: object,
    openings: Sequence[OpeningView],
    room_polygon: Sequence[Point2D],
) -> None:
    room_center = _polygon_center(room_polygon)
    for opening in openings:
        tangent, normal = _opening_vectors(opening=opening, room_center=room_center)
        if opening.kind == "door":
            _draw_door_opening(ax=ax, opening=opening, tangent=tangent, normal=normal)
        else:
            _draw_window_opening(ax=ax, opening=opening)
        center = Point2D(
            x=(opening.start.x + opening.end.x) / 2.0,
            y=(opening.start.y + opening.end.y) / 2.0,
        )
        label_normal = Point2D(x=-normal.x, y=-normal.y)
        label_center = Point2D(
            x=center.x + label_normal.x * 170.0,
            y=center.y + label_normal.y * 170.0,
        )
        is_window = opening.kind == "window"
        ax.text(
            label_center.x,
            label_center.y,
            "window" if is_window else "door",
            fontsize=6.4,
            fontweight="bold",
            color=WINDOW_LABEL_TEXT if is_window else DOOR_LABEL_TEXT,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": WINDOW_LABEL_FILL if is_window else DOOR_LABEL_FILL,
                "edgecolor": WINDOW_LABEL_STROKE if is_window else DOOR_LABEL_STROKE,
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=30,
        )


def _draw_door_opening(
    *,
    ax: object,
    opening: OpeningView,
    tangent: Point2D,
    normal: Point2D,
) -> None:
    segment_length = _distance(opening.start, opening.end)
    radius = _clamp(segment_length * 0.86, 420.0, 900.0)
    hinge = opening.start
    arc_points = _arc_points(
        center=hinge,
        radius=radius,
        start_angle=_angle(tangent),
        end_angle=_angle(normal),
    )
    leaf_end = Point2D(
        x=hinge.x + normal.x * radius,
        y=hinge.y + normal.y * radius,
    )
    _plot_segment(
        ax=ax,
        start=opening.start,
        end=opening.end,
        color=ROOM_FILL,
        linewidth=8.0,
        zorder=8,
    )
    _plot_segment(
        ax=ax,
        start=hinge,
        end=leaf_end,
        color=DOOR_STROKE,
        linewidth=1.6,
        zorder=10,
        alpha=0.78,
    )
    ax.plot(
        [point.x for point in arc_points],
        [point.y for point in arc_points],
        color=DOOR_STROKE,
        linewidth=1.1,
        linestyle=(0, (3, 3)),
        alpha=0.62,
        zorder=10,
    )
    _plot_segment(
        ax=ax,
        start=opening.start,
        end=opening.end,
        color=DOOR_STROKE,
        linewidth=3.0,
        zorder=11,
    )


def _draw_window_opening(*, ax: object, opening: OpeningView) -> None:
    _plot_segment(
        ax=ax,
        start=opening.start,
        end=opening.end,
        color=ROOM_FILL,
        linewidth=8.0,
        zorder=8,
    )
    _plot_segment(
        ax=ax,
        start=opening.start,
        end=opening.end,
        color=WINDOW_STROKE,
        linewidth=3.6,
        zorder=10,
        linestyle=(0, (5, 4)),
    )
    _plot_segment(
        ax=ax,
        start=opening.start,
        end=opening.end,
        color="#ffffff",
        linewidth=0.9,
        zorder=11,
        alpha=0.82,
    )


def _plot_segment(
    *,
    ax: object,
    start: Point2D,
    end: Point2D,
    color: str,
    linewidth: float,
    zorder: int,
    alpha: float = 1.0,
    linestyle: object = "-",
) -> None:
    ax.plot(
        [start.x, end.x],
        [start.y, end.y],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="butt",
        dash_capstyle="butt",
        alpha=alpha,
        zorder=zorder,
    )


def _draw_front_marker(*, ax: object, row: ObjectView) -> None:
    direction = _object_direction(row)
    if direction is None:
        return

    center = _polygon_center(row.polygon)
    edge_distance = _ray_distance_to_polygon_edge(
        center=center,
        direction=direction,
        polygon=row.polygon,
    )
    if edge_distance is None or edge_distance <= 1e-6:
        return

    tip_distance = edge_distance * 0.76
    tip = Point2D(
        x=center.x + direction.x * tip_distance,
        y=center.y + direction.y * tip_distance,
    )
    for color, linewidth, mutation_scale, zorder in (
        ("#ffffff", 4.2, 13.0, 16),
        ("#34444d", 1.8, 10.0, 17),
    ):
        ax.annotate(
            "",
            xy=(tip.x, tip.y),
            xytext=(center.x, center.y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": linewidth,
                "mutation_scale": mutation_scale,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=zorder,
        )


def _object_direction(row: ObjectView) -> Point2D | None:
    front = _vector(row.raw.get("front_world"))
    if front is not None:
        return front

    front_side = _first_string(row.raw, "front_side_world", "facing").lower()
    if front_side in _DIRECTION_SIDE_VECTORS:
        return _DIRECTION_SIDE_VECTORS[front_side]

    if not any(
        key in row.raw
        for key in (
            "angle",
            "angle_degrees",
            "rotation",
            "rotation_ccw",
            "rotation_degrees",
            "theta",
        )
    ):
        return None
    radians = math.radians(row.rotation_degrees)
    return Point2D(x=math.sin(radians), y=math.cos(radians))


def _vector(value: object) -> Point2D | None:
    item = _mapping(value)
    dx = _number(item.get("dx"))
    dy = _number(item.get("dy"))
    if dx is None or dy is None or abs(dx) + abs(dy) <= 1e-6:
        return None
    return _normalized(Point2D(x=dx, y=dy))


def _ray_distance_to_polygon_edge(
    *,
    center: Point2D,
    direction: Point2D,
    polygon: Sequence[Point2D],
) -> float | None:
    if not polygon:
        return None

    distances: list[float] = []
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        edge = Point2D(x=end.x - start.x, y=end.y - start.y)
        denominator = _cross(direction, edge)
        if abs(denominator) <= 1e-9:
            continue
        offset = Point2D(x=start.x - center.x, y=start.y - center.y)
        distance = _cross(offset, edge) / denominator
        edge_ratio = _cross(offset, direction) / denominator
        if distance > 1e-6 and -1e-6 <= edge_ratio <= 1.0 + 1e-6:
            distances.append(distance)
    return min(distances) if distances else None


def _cross(first: Point2D, second: Point2D) -> float:
    return first.x * second.y - first.y * second.x


def _draw_object_label(*, ax: object, row: ObjectView) -> None:
    min_x, min_y, max_x, max_y = _polygon_bounds(row.polygon)
    width = max_x - min_x
    height = max_y - min_y
    area = max(0.0, width * height)
    if area < 55_000.0:
        return

    center = _polygon_center(row.polygon)
    label = _display_type(row.object_type)
    font_size = 6.4 if min(width, height) < 450.0 else 7.4
    ax.text(
        center.x,
        center.y,
        label,
        ha="center",
        va="center",
        fontsize=font_size,
        color="#1f1f1f",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#ffffff",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        clip_on=True,
        zorder=18,
    )


def _apply_axes_layout(
    *,
    ax: object,
    extents: tuple[float, float, float, float],
) -> None:
    min_x, min_y, max_x, max_y = extents
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    padding = max(180.0, max(width, height) * 0.055)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _layout_extents(layout: LayoutView) -> tuple[float, float, float, float]:
    points = list(layout.room_polygon)
    for opening in layout.openings:
        points.extend((opening.start, opening.end))
    for row in layout.objects:
        points.extend(row.polygon)
    if not points:
        return (0.0, 0.0, 1000.0, 1000.0)
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    if max_x <= min_x:
        max_x = min_x + 1000.0
    if max_y <= min_y:
        max_y = min_y + 1000.0
    return (min_x, min_y, max_x, max_y)


def _figure_size(extents: tuple[float, float, float, float]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = extents
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    aspect = width / height
    max_side = 8.4
    min_side = 4.8
    if aspect >= 1.0:
        return (max_side, max(min_side, max_side / aspect))
    return (max(min_side, max_side * aspect), max_side)


def _load_case_index(path: Path) -> dict[str, Mapping[str, object]]:
    if not path.is_file():
        _LOGGER.warning("Cases file not found: %s", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, Mapping[str, object]] = {}
    for row in _sequence(payload):
        item = _mapping(row)
        case_id = _first_string(item, "case_id")
        if case_id:
            index[case_id] = item
    return index


def _case_from_input(case_dir: Path, case_id: str) -> Mapping[str, object]:
    input_path = case_dir / "input.json"
    if not input_path.is_file():
        raise ValueError(f"No benchmark case metadata for {case_id}")
    payload = _load_json_mapping(input_path)
    user_input = _mapping(payload.get("user_input"))
    polygon = list(_sequence(user_input.get("shape_points")))
    doors = list(_sequence(payload.get("doors")))
    windows = list(_sequence(payload.get("windows")))
    return {
        "case_id": case_id,
        "room": {
            "polygon_mm": polygon,
            "openings": {
                "doors": doors,
                "windows": windows,
            },
        },
    }


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = _mapping(payload)
    if not item:
        raise ValueError(f"Expected a JSON object in {path}")
    return item


def _infer_normalizer(*, system_folder: str, case_dir: Path) -> str:
    text = f"{system_folder} {case_dir.name}".lower()
    if "flairgpt" in text:
        return "flairgpt"
    if "idesign" in text:
        return "idesign"
    if "tknt" in text:
        return "tknt"
    raise ValueError(f"Cannot infer normalizer for {case_dir}")


def _system_id_from_dir_name(case_dir_name: str, system_folder: str) -> str:
    if "__" not in case_dir_name:
        return system_folder
    suffix = case_dir_name.split("__", 1)[1].strip()
    return suffix or system_folder


def _case_id_from_dir_name(case_dir_name: str) -> str:
    return case_dir_name.split("__", 1)[0].strip()


def _object_layer(row: ObjectView) -> int:
    if _has_any_token(row.object_type, _FLOOR_LAYER_TOKENS):
        return 0
    if _has_any_token(row.object_type, _SOFT_LAYER_TOKENS):
        return 2
    return 1


def _object_color(row: ObjectView) -> str:
    raw_color = _first_string(row.raw, "color_hex", "color", "fill")
    if _is_hex_color(raw_color):
        return raw_color
    display_type = _display_type(row.object_type)
    if display_type in _TYPE_COLORS:
        return _TYPE_COLORS[display_type]
    for key, color in _TYPE_COLORS.items():
        if _type_matches(display_type, key):
            return color
    return "#cfcac2"


def _display_type(object_type: str) -> str:
    normalized = object_type.strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_\d+$", "", normalized)


def _type_matches(object_type: str, token: str) -> bool:
    return (
        object_type == token
        or object_type.startswith(f"{token}_")
        or object_type.endswith(f"_{token}")
        or f"_{token}_" in object_type
    )


def _has_any_token(object_type: str, tokens: Sequence[str]) -> bool:
    display_type = _display_type(object_type)
    return any(_type_matches(display_type, token) for token in tokens)


def _is_hex_color(value: str) -> bool:
    return re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None


def _opening_kind(value: object) -> OpeningKind | None:
    if value == "door":
        return "door"
    if value == "window":
        return "window"
    return None


def _point(value: object) -> Point2D | None:
    item = _mapping(value)
    x = _number(item.get("x"))
    y = _number(item.get("y"))
    if x is None or y is None:
        return None
    return Point2D(x=x, y=y)


def _xy_rows(points: Sequence[Point2D]) -> list[tuple[float, float]]:
    return [(point.x, point.y) for point in points]


def _polygon_center(points: Sequence[Point2D]) -> Point2D:
    if not points:
        return Point2D(x=0.0, y=0.0)
    return Point2D(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def _polygon_bounds(points: Sequence[Point2D]) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 1.0, 1.0)
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _opening_vectors(
    *,
    opening: OpeningView,
    room_center: Point2D,
) -> tuple[Point2D, Point2D]:
    tangent = _normalized(
        Point2D(
            x=opening.end.x - opening.start.x,
            y=opening.end.y - opening.start.y,
        )
    )
    center = Point2D(
        x=(opening.start.x + opening.end.x) / 2.0,
        y=(opening.start.y + opening.end.y) / 2.0,
    )
    first_normal = Point2D(x=-tangent.y, y=tangent.x)
    toward_center = Point2D(
        x=room_center.x - center.x,
        y=room_center.y - center.y,
    )
    dot = first_normal.x * toward_center.x + first_normal.y * toward_center.y
    if dot >= 0:
        return tangent, first_normal
    return tangent, Point2D(x=-first_normal.x, y=-first_normal.y)


def _normalized(vector: Point2D) -> Point2D:
    length = (vector.x * vector.x + vector.y * vector.y) ** 0.5
    if length <= 1e-6:
        return Point2D(x=1.0, y=0.0)
    return Point2D(x=vector.x / length, y=vector.y / length)


def _arc_points(
    *,
    center: Point2D,
    radius: float,
    start_angle: float,
    end_angle: float,
) -> tuple[Point2D, ...]:
    delta = _normalized_angle_delta(end_angle - start_angle)
    points: list[Point2D] = []
    for index in range(13):
        angle = start_angle + delta * index / 12.0
        points.append(
            Point2D(
                x=center.x + math.cos(angle) * radius,
                y=center.y + math.sin(angle) * radius,
            )
        )
    return tuple(points)


def _angle(vector: Point2D) -> float:
    return math.atan2(vector.y, vector.x)


def _normalized_angle_delta(value: float) -> float:
    delta = value
    while delta > math.pi:
        delta -= math.tau
    while delta < -math.pi:
        delta += math.tau
    return delta


def _distance(first: Point2D, second: Point2D) -> float:
    return ((second.x - first.x) ** 2 + (second.y - first.y) ** 2) ** 0.5


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _floor_to_grid(value: float) -> float:
    return (value // GRID_SIZE_MM) * GRID_SIZE_MM


def _ceil_to_grid(value: float) -> float:
    return ((value + GRID_SIZE_MM - 1.0) // GRID_SIZE_MM) * GRID_SIZE_MM


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _first_string(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return ""


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
