from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout.tools import build_absolute_layout
from agent.stylist import Stylist
from test_agent.render_layout import _plot_polygon  # reuse plot util
import matplotlib.pyplot as plt


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _render_from_stylist(result: dict[str, Any], out_path: Path) -> None:
    room = result.get("room") or {}
    room_poly = room.get("polygon_ccw") or []
    obstacles = room.get("obstacles") or []
    surfaces = room.get("surfaces") or {}
    opening_colors = room.get("opening_colors") or {}
    openings = room.get("openings") or {}

    fig, ax = plt.subplots(figsize=(10, 7))

    # room
    if room_poly:
        pts = [(int(p["x"]), int(p["y"])) for p in room_poly]
        wall_color = surfaces.get("wall_color_hex", "#f7f7f7")
        _plot_polygon(ax, pts, edge="#111111", face=wall_color, alpha=1.0, lw=2.0)

    # obstacles
    for ob in obstacles:
        pts = ob.get("polygon_ccw") or []
        if len(pts) < 3:
            continue
        ob_pts = [(int(p["x"]), int(p["y"])) for p in pts]
        _plot_polygon(ax, ob_pts, edge="#aa0000", face="#ffcccc", alpha=0.8, lw=1.0)

    # objects (render rugs first as underlay)
    objects = result.get("objects", []) or []
    rugs: list[dict] = []
    others: list[dict] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_type = str(obj.get("object_type") or "").lower()
        if "rug" in obj_type:
            rugs.append(obj)
        else:
            others.append(obj)

    def _draw_objects(objs: list[dict], alpha: float) -> None:
        for obj in objs:
            poly = obj.get("polygon_ccw") or []
            if len(poly) < 3:
                continue
            pts = [(int(p["x"]), int(p["y"])) for p in poly]
            color = obj.get("color_hex") or "#dddddd"
            _plot_polygon(ax, pts, edge="#111111", face=color, alpha=alpha, lw=1.2)
            label = obj.get("object_type") or obj.get("instance_id") or "obj"
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            ax.text(cx, cy, str(label), fontsize=7, ha="center", va="center", color="#000000")

    _draw_objects(rugs, alpha=0.5)
    _draw_objects(others, alpha=0.9)

    # opening segments (door/window)
    def _draw_segments(items, color):
        for it in items or []:
            seg = it.get("segment_mm") or []
            if len(seg) != 2:
                continue
            x1, y1 = int(seg[0]["x"]), int(seg[0]["y"])
            x2, y2 = int(seg[1]["x"]), int(seg[1]["y"])
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)

    door_colors = {d.get("id"): d.get("color_hex") for d in (opening_colors.get("doors") or [])}
    window_colors = {w.get("id"): w.get("color_hex") for w in (opening_colors.get("windows") or [])}

    for d in openings.get("doors", []) or []:
        color = door_colors.get(d.get("id"), "#8b5a2b")
        _draw_segments([d], color)
    for w in openings.get("windows", []) or []:
        color = window_colors.get(w.get("id"), "#6aaed6")
        _draw_segments([w], color)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title("Stylist Render Preview")
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="BuildAbsoluteLayout -> Stylist -> Renderer")
    parser.add_argument(
        "--room",
        default=str(REPO_ROOT / "test_agent" / "result" / "room_interpreter_output.json"),
        help="RoomInterpreter output JSON",
    )
    parser.add_argument(
        "--clusters",
        default=str(REPO_ROOT / "test_agent" / "result" / "cluster_outline_output.json"),
        help="Cluster outlines JSON",
    )
    parser.add_argument(
        "--placer",
        default=str(REPO_ROOT / "test_agent" / "result" / "cluster_placer_output.json"),
        help="Cluster placer output JSON",
    )
    parser.add_argument(
        "--tenant",
        default="",
        help="tenant_id for inventory lookup",
    )
    parser.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "test_agent" / "result" / "stylist_output.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-img",
        default=str(REPO_ROOT / "test_agent" / "result" / "stylist_render.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    room_model = _load_json(Path(args.room))
    clusters_outlines = _load_json(Path(args.clusters))
    placer = _load_json(Path(args.placer))

    layout = build_absolute_layout(
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        cluster_transforms=placer.get("cluster_transforms", []),
    )

    stylist = Stylist()
    result = stylist.generate(
        layout_json=layout,
        user_context_json={"tenant_id": args.tenant} if args.tenant else {},
        tenant_id=args.tenant if args.tenant else None,
    )

    _save_json(Path(args.out_json), result)
    _render_from_stylist(result, Path(args.out_img))

    print("Saved stylist JSON to:", args.out_json)
    print("Saved render PNG to:", args.out_img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
