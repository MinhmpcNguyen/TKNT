from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _unwrap(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    parsed = obj.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    raw = obj.get("raw")
    if isinstance(raw, dict):
        return raw
    return obj


def _normalize_clusters_outlines(data: Any) -> Dict[str, Dict[str, Any]]:
    data = _unwrap(data)
    if isinstance(data, dict) and "cluster_footprint" in data:
        cid = data.get("cluster_id", "cluster_0")
        return {str(cid): data}
    if isinstance(data, dict) and isinstance(data.get("clusters"), list):
        out: Dict[str, Dict[str, Any]] = {}
        for c in data["clusters"]:
            if isinstance(c, dict) and isinstance(c.get("cluster_id"), str):
                out[c["cluster_id"]] = c
        return out
    if isinstance(data, dict):
        # assume already keyed by cluster_id
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _normalize_transforms(data: Any) -> Dict[str, Dict[str, Any]]:
    data = _unwrap(data)
    if not isinstance(data, dict):
        return {}
    transforms = data.get("cluster_transforms")
    if not isinstance(transforms, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for t in transforms:
        if not isinstance(t, dict):
            continue
        cid = t.get("cluster_id")
        if isinstance(cid, str):
            out[cid] = t
    return out


def _rotate_ccw_90s(x: int, y: int, rot: int) -> Tuple[int, int]:
    r = rot % 360
    if r == 0:
        return x, y
    if r == 90:
        return -y, x
    if r == 180:
        return -x, -y
    if r == 270:
        return y, -x
    return x, y


def _transform_polygon(poly: List[Dict[str, int]], x: int, y: int, rot: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for p in poly:
        px, py = int(p["x"]), int(p["y"])
        rx, ry = _rotate_ccw_90s(px, py, rot)
        pts.append((rx + x, ry + y))
    return pts


def _transform_rect(rect: Dict[str, Any], x: int, y: int, rot: int) -> List[Tuple[int, int]]:
    rx = int(rect.get("x", 0))
    ry = int(rect.get("y", 0))
    w = int(rect.get("w", 0))
    h = int(rect.get("h", 0))
    if w <= 0 or h <= 0:
        return []
    corners = [
        {"x": rx, "y": ry},
        {"x": rx + w, "y": ry},
        {"x": rx + w, "y": ry + h},
        {"x": rx, "y": ry + h},
    ]
    return _transform_polygon(corners, x, y, rot)


def _poly_centroid(pts: List[Tuple[int, int]]) -> Tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    x_sum = sum(p[0] for p in pts)
    y_sum = sum(p[1] for p in pts)
    return (x_sum / len(pts), y_sum / len(pts))


def _plot_polygon(ax, pts: List[Tuple[int, int]], *, edge: str, face: str, alpha: float, lw: float = 1.0):
    if not pts:
        return
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.fill(xs, ys, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render room + clusters outlines + placements")
    parser.add_argument(
        "--room",
        default="/Users/Yuki/TKNT/test_agent/result/room_interpreter_output.json",
        help="RoomInterpreter output JSON",
    )
    parser.add_argument(
        "--clusters",
        default="/Users/Yuki/TKNT/test_agent/result/cluster_outline_output.json",
        help="Cluster outlines JSON",
    )
    parser.add_argument(
        "--placer",
        default="/Users/Yuki/TKNT/test_agent/result/cluster_placer_output.json",
        help="Cluster placer output JSON",
    )
    parser.add_argument(
        "--out",
        default="/Users/Yuki/TKNT/test_agent/result/layout_preview.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--no-objects",
        action="store_true",
        help="Disable rendering individual objects (rects) within each cluster.",
    )
    parser.add_argument(
        "--no-outlines",
        action="store_true",
        help="Disable rendering cluster outline polygons.",
    )
    args = parser.parse_args()

    room_model = _unwrap(_load_json(Path(args.room)))
    clusters_outlines = _normalize_clusters_outlines(_load_json(Path(args.clusters)))
    transforms = _normalize_transforms(_load_json(Path(args.placer)))

    room_poly = (room_model.get("room") or {}).get("polygon_ccw") or []
    if not room_poly:
        raise ValueError("room polygon_ccw missing")

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Room
    room_pts = [(int(p["x"]), int(p["y"])) for p in room_poly]
    _plot_polygon(ax, room_pts, edge="#222222", face="#f8f8f8", alpha=1.0, lw=2.0)

    # Obstacles
    for ob in (room_model.get("obstacles") or []):
        pts = ob.get("polygon_ccw") or []
        if len(pts) < 3:
            continue
        ob_pts = [(int(p["x"]), int(p["y"])) for p in pts]
        _plot_polygon(ax, ob_pts, edge="#aa0000", face="#ffcccc", alpha=0.8, lw=1.0)

    show_objects = not args.no_objects
    show_outlines = not args.no_outlines

    # Cluster outlines with transforms
    palette = [
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#fd8d3c",
        "#fdae6b",
        "#74c476",
        "#a1d99b",
        "#9e9ac8",
    ]
    idx = 0
    for cid, cinfo in clusters_outlines.items():
        t = transforms.get(cid)
        if not isinstance(t, dict):
            continue
        x = int(t.get("x", 0))
        y = int(t.get("y", 0))
        rot = int(t.get("rot", 0)) % 360

        outlines = ((cinfo.get("cluster_footprint") or {}).get("outline_polygons_ccw")
                    or cinfo.get("outline_polygons_ccw")
                    or [])
        if not isinstance(outlines, list):
            continue

        color = palette[idx % len(palette)]
        idx += 1

        if show_outlines:
            for poly in outlines:
                if not isinstance(poly, list) or len(poly) < 3:
                    continue
                pts = _transform_polygon(poly, x, y, rot)
                _plot_polygon(ax, pts, edge="#333333", face=color, alpha=0.35, lw=1.0)
                cx, cy = _poly_centroid(pts)
                ax.text(cx, cy, cid, fontsize=9, ha="center", va="center", color="#111111")

        if show_objects:
            rects = (cinfo.get("cluster_footprint") or {}).get("rects") or []
            if isinstance(rects, list):
                for r in rects:
                    if not isinstance(r, dict):
                        continue
                    r_pts = _transform_rect(r, x, y, rot)
                    if not r_pts:
                        continue
                    _plot_polygon(ax, r_pts, edge="#111111", face="#f7f7f7", alpha=0.9, lw=1.2)
                    obj_id = r.get("id")
                    if isinstance(obj_id, str):
                        ox, oy = _poly_centroid(r_pts)
                        ax.text(ox, oy, obj_id, fontsize=7, ha="center", va="center", color="#000000")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title("Macro Layout Preview")
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved layout preview to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
