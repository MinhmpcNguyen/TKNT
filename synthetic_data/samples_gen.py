import json
import math
import random
from typing import Dict, List, Tuple

Point = Dict[str, float]

# ---- Example asset pools (replace by DB export) ----
DOOR_ASSETS = [
    {"asset_id": "door_900", "leaf_width_mm": 900, "open_angle_deg": 90},
    {"asset_id": "door_800", "leaf_width_mm": 800, "open_angle_deg": 90},
]
WINDOW_ASSETS = [
    {"asset_id": "win_1200", "width_mm": 1200},
    {"asset_id": "win_1500", "width_mm": 1500},
    {"asset_id": "win_1800", "width_mm": 1800},
]

ROOM_TYPES = ["bedroom", "living_room", "studio", "office", "kitchen"]
STYLES = ["modern", "japandi", "scandinavian", "minimal", "industrial", "classic"]
DIRECTIONS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]

SPECIAL_BANK = {
    "bedroom": [
        "ưu tiên bàn làm việc",
        "cần nhiều lưu trữ",
        "có 2 người dùng",
        "cần góc đọc sách",
    ],
    "living_room": [
        "ưu tiên chỗ ngồi nhiều",
        "có TV lớn",
        "tối giản đồ",
        "ưu tiên không gian trống",
    ],
    "studio": [
        "kết hợp ngủ và làm việc",
        "ưu tiên không gian trống",
        "ngân sách tiết kiệm",
    ],
    "office": [
        "cần 2 bàn làm việc",
        "cần tủ tài liệu lớn",
        "ưu tiên ánh sáng tự nhiên",
    ],
    "kitchen": ["cần nhiều mặt bàn", "ưu tiên công năng", "cần nhiều tủ bếp"],
}


def rect_from_area(area_m2: float, rnd: random.Random) -> List[Tuple[float, float]]:
    area_mm2 = area_m2 * 1_000_000
    w = rnd.randint(2800, 6500)
    d = max(2000, int(area_mm2 / w))
    return [(0, 0), (w, 0), (w, d), (0, d)]


def edge_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def lerp(
    a: Tuple[float, float], b: Tuple[float, float], t: float
) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def place_segment_on_edge(
    a, b, seg_len, center_t
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # place segment of length seg_len centered at parameter center_t along edge AB
    L = edge_length(a, b)
    if seg_len > L:
        raise ValueError("segment longer than edge")
    half_t = (seg_len / 2) / L
    t1 = max(0.0, center_t - half_t)
    t2 = min(1.0, center_t + half_t)
    p1 = lerp(a, b, t1)
    p2 = lerp(a, b, t2)
    return p1, p2


def pick_wall_for_direction(rect_pts, direction: str):
    # rect_pts: [(0,0),(W,0),(W,D),(0,D)]
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = rect_pts
    if direction.startswith("N"):  # N, NE, NW
        return ((x0, y0), (x1, y1))  # top
    if direction.startswith("S"):
        return ((x3, y3), (x2, y2))  # bottom
    if direction.startswith("E"):
        return ((x1, y1), (x2, y2))  # right
    return ((x0, y0), (x3, y3))  # left


def choose_window_assets(
    n: int, wall_len: float, rnd: random.Random, corner_margin=200, gap=300
):
    usable = wall_len - 2 * corner_margin
    if usable <= 0:
        return None
    # pick widths that fit: greedy choose smaller if needed
    widths = []
    for _ in range(n):
        cand = rnd.choice(WINDOW_ASSETS)
        widths.append(cand["width_mm"])

    # shrink if overflow by replacing with smaller assets
    def total(wlist):
        return sum(wlist) + gap * (len(wlist) - 1) if wlist else 0

    widths.sort(reverse=True)
    while total(widths) > usable and widths:
        # replace the largest with the smallest available
        widths[0] = min(a["width_mm"] for a in WINDOW_ASSETS)
        widths.sort(reverse=True)
        # if still too big, drop a window (for conflicting/needs_info cases you might keep and mark impossible)
        if total(widths) > usable and len(widths) > 1:
            widths.pop(0)
            widths.sort(reverse=True)
        else:
            break
    return widths


def build_room_geometry(
    seed: int, area_m2: float, windows_n: int, window_direction: str
):
    rnd = random.Random(seed)
    rect = rect_from_area(area_m2, rnd)
    W = rect[1][0]
    D = rect[2][1]

    corner_margin = 200
    gap = 300

    # Windows
    wall_a, wall_b = pick_wall_for_direction(rect, window_direction)
    wall_len = edge_length(wall_a, wall_b)

    widths = choose_window_assets(windows_n, wall_len, rnd, corner_margin, gap)
    windows = []
    if widths is None:
        widths = []
    if widths:
        usable = wall_len - 2 * corner_margin
        total_w = sum(widths) + gap * (len(widths) - 1)
        start_offset = (usable - total_w) / 2  # center them
        # convert offset along wall into t
        L = wall_len
        offset = corner_margin + start_offset
        for i, wmm in enumerate(widths):
            center_offset = offset + wmm / 2
            center_t = center_offset / L
            p1, p2 = place_segment_on_edge(wall_a, wall_b, wmm, center_t)
            asset_id = min(WINDOW_ASSETS, key=lambda a: abs(a["width_mm"] - wmm))[
                "asset_id"
            ]
            windows.append(
                {
                    "id": f"win_{i + 1}",
                    "asset_id": asset_id,
                    "segment": {
                        "p1": {"x": round(p1[0], 2), "y": round(p1[1], 2)},
                        "p2": {"x": round(p2[0], 2), "y": round(p2[1], 2)},
                    },
                }
            )
            offset += wmm + gap

    # Door: place on a different wall (prefer bottom unless windows already there)
    walls = {
        "top": ((0, 0), (W, 0)),
        "bottom": ((0, D), (W, D)),
        "left": ((0, 0), (0, D)),
        "right": ((W, 0), (W, D)),
    }
    win_wall = (
        "top"
        if wall_a[1] == 0 and wall_b[1] == 0
        else "bottom"
        if wall_a[1] == D and wall_b[1] == D
        else "left"
        if wall_a[0] == 0 and wall_b[0] == 0
        else "right"
    )
    door_wall_name = rnd.choice([k for k in walls.keys() if k != win_wall])
    da, db = walls[door_wall_name]
    dlen = edge_length(da, db)

    # pick a door asset that fits
    possible = [
        a for a in DOOR_ASSETS if a["leaf_width_mm"] <= dlen - 2 * corner_margin
    ]
    door_asset = rnd.choice(possible) if possible else rnd.choice(DOOR_ASSETS)
    leaf_w = door_asset["leaf_width_mm"]

    center_t = 0.5
    p1, p2 = place_segment_on_edge(da, db, leaf_w, center_t)
    hinge_side = rnd.choice(["left", "right"])

    doors = [
        {
            "id": "door_1",
            "asset_id": door_asset["asset_id"],
            "leaf_width_mm": leaf_w,
            "open_angle_deg": door_asset.get("open_angle_deg", 90),
            "hinge_side": hinge_side,
            "open_direction": "inward",
            "segment": {
                "p1": {"x": round(p1[0], 2), "y": round(p1[1], 2)},
                "p2": {"x": round(p2[0], 2), "y": round(p2[1], 2)},
            },
        }
    ]

    room_polygon_mm = [{"x": float(x), "y": float(y)} for x, y in rect]

    return {"room_polygon_mm": room_polygon_mm, "doors": doors, "windows": windows}


def make_feng_shui(seed: int) -> str:
    rnd = random.Random(seed)
    menh = rnd.choice(["Kim", "Mộc", "Thủy", "Hỏa", "Thổ"])
    # tuổi chỉ để làm chuỗi; không cần chính xác
    birth_year = rnd.randint(1985, 2008)
    notes = rnd.choice(
        [
            "tránh giường thẳng cửa",
            "ưu tiên lối đi thoáng",
            "bàn làm việc nên nhìn ra cửa",
            "tránh đặt gương đối diện giường",
        ]
    )
    return f"menh={menh}; nam_sinh={birth_year}; ghi_chu={notes}"


def make_description(room_type, area, style, windows, direction, special):
    parts = [f"Thiết kế {room_type}.", f"Diện tích khoảng {area:.1f}m2."]
    if style:
        parts.append(f"Phong cách {style}.")
    parts.append(f"Có {windows} cửa sổ hướng {direction}.")
    if special:
        parts.append(f"Yêu cầu thêm: {special}.")
    return " ".join(parts)


def synthesize_one(seed: int):
    rnd = random.Random(seed)
    room_type = rnd.choice(ROOM_TYPES)
    # area ranges
    area_ranges = {
        "bedroom": (9, 18),
        "living_room": (12, 28),
        "studio": (18, 35),
        "office": (6, 14),
        "kitchen": (6, 16),
    }
    area = rnd.uniform(*area_ranges[room_type])
    style = rnd.choice(STYLES)
    windows_n = rnd.choice([0, 1, 1, 2, 2, 3])
    direction = rnd.choice(DIRECTIONS)
    special = rnd.choice(SPECIAL_BANK[room_type]) if rnd.random() < 0.8 else None
    feng = make_feng_shui(seed) if rnd.random() < 0.6 else None
    height = float(rnd.randint(2400, 3200))

    room_geom = build_room_geometry(seed, area, windows_n, direction)

    user_input = {
        "description": make_description(
            room_type, area, style, windows_n, direction, special
        ),
        "room_type": room_type,
        "floor_area_m2": round(area, 2),
        "height": height,
        "shape_points": [
            {"x": p["x"], "y": p["y"]} for p in room_geom["room_polygon_mm"]
        ],
        "location_floor": rnd.randint(1, 30),
        "bathrooms": rnd.choice([0, 0, 1, 1, 2]),
        "windows": windows_n,
        "window_direction": direction,
        "style": style,
        "special_description": special,
        "feng_shui": feng,
    }

    # tags: normal / ambiguous / conflicting
    r = rnd.random()
    tag = "normal"
    if r < 0.20:
        tag = "ambiguous"
        # drop 1-2 fields but keep room_geom for benchmarking
        for k in rnd.sample(
            ["shape_points", "style", "window_direction"], k=rnd.randint(1, 2)
        ):
            user_input[k] = None
    elif r < 0.40:
        tag = "conflicting"
        # small room but heavy asks
        user_input["floor_area_m2"] = round(rnd.uniform(8.0, 10.0), 2)
        user_input["special_description"] = "cần nhiều đồ và 2 khu vực làm việc"
        user_input["description"] = make_description(
            room_type,
            user_input["floor_area_m2"],
            user_input.get("style"),
            windows_n,
            direction,
            user_input["special_description"],
        )

    return {
        "user_input": user_input,
        "room_geometry": room_geom,
        "meta": {"seed": seed, "tag": tag},
    }


def generate_jsonl(path: str, n: int, base_seed: int = 2026):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            sample = synthesize_one(base_seed + i)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generate_jsonl("dataset.jsonl", n=500, base_seed=2026)
    print("Wrote dataset.jsonl")
