import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- Embedded JSON ---
DATA = {
    "status": "OK",
    "room": {
        "room_id": "room_1",
        "room_type": "bedroom",
        "polygon_ccw": [
            {"x": 0, "y": 0},
            {"x": 6000, "y": 0},
            {"x": 5967, "y": 2475},
            {"x": 3496, "y": 2514},
            {"x": 3482, "y": 3477},
            {"x": 17, "y": 3477},
        ],
        "obstacles": [
            {
                "id": "door_swing_1",
                "type": "door_swing",
                "polygon_ccw": [
                    {"x": 15, "y": 2158},
                    {"x": 8, "y": 1158},
                    {"x": 608, "y": 1158},
                    {"x": 615, "y": 2158},
                ],
                "hard": True,
            },
            {
                "id": "window_clearance_1",
                "type": "window_clearance",
                "polygon_ccw": [
                    {"x": 2000, "y": 0},
                    {"x": 4000, "y": 0},
                    {"x": 4000, "y": 100},
                    {"x": 2000, "y": 100},
                ],
                "hard": True,
            },
        ],
        "openings": {
            "doors": [
                {
                    "id": "door_1",
                    "segment_mm": [{"x": 15, "y": 2158}, {"x": 8, "y": 1158}],
                }
            ],
            "windows": [
                {
                    "id": "window_1",
                    "segment_mm": [{"x": 2000, "y": 0}, {"x": 4000, "y": 0}],
                }
            ],
        },
        "surfaces": {
            "wall_color_hex": "#F3E7D8",
            "floor_color_hex": "#C28E63",
            "ceiling_color_hex": "#F3E7D8",
        },
        "opening_colors": {
            "doors": [{"id": "door_1", "color_hex": "#A57C67"}],
            "windows": [{"id": "window_1", "color_hex": "#A57C67"}],
        },
    },
    "objects": [
        {
            "instance_id": "bed",
            "object_type": "bed",
            "source": "existing",
            "cluster_id": "sleep_area",
            "polygon_ccw": [
                {"x": 3240, "y": 800},
                {"x": 3240, "y": 2760},
                {"x": 1550, "y": 2760},
                {"x": 1550, "y": 800},
            ],
            "bbox": {"min_x": 1550, "min_y": 800, "max_x": 3240, "max_y": 2760},
            "color_hex": "#DAB997",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "bench",
            "object_type": "bench",
            "source": "existing",
            "cluster_id": "sleep_area",
            "polygon_ccw": [
                {"x": 1550, "y": 1300},
                {"x": 1550, "y": 2230},
                {"x": 1150, "y": 2230},
                {"x": 1150, "y": 1300},
            ],
            "bbox": {"min_x": 1150, "min_y": 1300, "max_x": 1550, "max_y": 2230},
            "color_hex": "#B98A61",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "nightstand",
            "object_type": "nightstand",
            "source": "existing",
            "cluster_id": "sleep_area",
            "polygon_ccw": [
                {"x": 3600, "y": 1550},
                {"x": 3600, "y": 1980},
                {"x": 3250, "y": 1980},
                {"x": 3250, "y": 1550},
            ],
            "bbox": {"min_x": 3250, "min_y": 1550, "max_x": 3600, "max_y": 1980},
            "color_hex": "#7B5A43",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "dresser",
            "object_type": "dresser",
            "source": "existing",
            "cluster_id": "storage_area",
            "polygon_ccw": [
                {"x": 2180, "y": -50},
                {"x": 2660, "y": -50},
                {"x": 2660, "y": 830},
                {"x": 2180, "y": 830},
            ],
            "bbox": {"min_x": 2180, "min_y": -50, "max_x": 2660, "max_y": 830},
            "color_hex": "#A57C67",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "storage_cabinet",
            "object_type": "storage_cabinet",
            "source": "existing",
            "cluster_id": "storage_area",
            "polygon_ccw": [
                {"x": 1450, "y": 50},
                {"x": 1860, "y": 50},
                {"x": 1860, "y": 660},
                {"x": 1450, "y": 660},
            ],
            "bbox": {"min_x": 1450, "min_y": 50, "max_x": 1860, "max_y": 660},
            "color_hex": "#B98A61",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "wardrobe",
            "object_type": "wardrobe",
            "source": "existing",
            "cluster_id": "storage_area",
            "polygon_ccw": [
                {"x": 2860, "y": 150},
                {"x": 4610, "y": 150},
                {"x": 4610, "y": 660},
                {"x": 2860, "y": 660},
            ],
            "bbox": {"min_x": 2860, "min_y": 150, "max_x": 4610, "max_y": 660},
            "color_hex": "#7B5A43",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "chair",
            "object_type": "chair",
            "source": "existing",
            "cluster_id": "work_area",
            "polygon_ccw": [
                {"x": 5100, "y": 1650},
                {"x": 5550, "y": 1650},
                {"x": 5550, "y": 2100},
                {"x": 5100, "y": 2100},
            ],
            "bbox": {"min_x": 5100, "min_y": 1650, "max_x": 5550, "max_y": 2100},
            "color_hex": "#A57C67",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "desk",
            "object_type": "desk",
            "source": "existing",
            "cluster_id": "work_area",
            "polygon_ccw": [
                {"x": 4760, "y": 1150},
                {"x": 5850, "y": 1150},
                {"x": 5850, "y": 1650},
                {"x": 4760, "y": 1650},
            ],
            "bbox": {"min_x": 4760, "min_y": 1150, "max_x": 5850, "max_y": 1650},
            "color_hex": "#DAB997",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "office_chair",
            "object_type": "office_chair",
            "source": "existing",
            "cluster_id": "work_area",
            "polygon_ccw": [
                {"x": 4200, "y": 1150},
                {"x": 4750, "y": 1150},
                {"x": 4750, "y": 1730},
                {"x": 4200, "y": 1730},
            ],
            "bbox": {"min_x": 4200, "min_y": 1150, "max_x": 4750, "max_y": 1730},
            "color_hex": "#B98A61",
            "material": None,
            "place_on": None,
        },
        {
            "instance_id": "rug_1",
            "object_type": "rug",
            "source": "inventory",
            "cluster_id": None,
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 2110, "y": 0},
                {"x": 2110, "y": 2520},
                {"x": 0, "y": 2520},
            ],
            "bbox": {"min_x": 0, "min_y": 0, "max_x": 2110, "max_y": 2520},
            "color_hex": "#8FA09A",
            "material": "metal",
            "place_on": {"target_instance_id": "floor", "method": "floor"},
        },
        {
            "instance_id": "bedside_lamp_1",
            "object_type": "bedside_lamp",
            "source": "inventory",
            "cluster_id": None,
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 200, "y": 0},
                {"x": 200, "y": 280},
                {"x": 0, "y": 280},
            ],
            "bbox": {"min_x": 0, "min_y": 0, "max_x": 200, "max_y": 280},
            "color_hex": "#B0B9BC",
            "material": "engineered_wood",
            "place_on": {"target_instance_id": "nightstand", "method": "on_top"},
        },
    ],
    "notes": [
        "Rotated storage_cabinet and dresser in storage_area to improve front access and open space clearance."
    ],
    "missing": [],
}


def poly(points):
    return [(p["x"], p["y"]) for p in points]


fig, ax = plt.subplots(figsize=(8, 6))

# Room
room_poly = Polygon(
    poly(DATA["room"]["polygon_ccw"]),
    closed=True,
    fill=False,
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(room_poly)

# Obstacles
for ob in DATA["room"].get("obstacles", []):
    p = Polygon(
        poly(ob["polygon_ccw"]),
        closed=True,
        fill=True,
        alpha=0.2,
        edgecolor="red",
        facecolor="red",
    )
    ax.add_patch(p)

# Objects
for obj in DATA["objects"]:
    p = Polygon(
        poly(obj["polygon_ccw"]),
        closed=True,
        fill=True,
        alpha=0.5,
        edgecolor="gray",
        facecolor=obj["color_hex"],
    )
    ax.add_patch(p)
    cx = sum(pt["x"] for pt in obj["polygon_ccw"]) / len(obj["polygon_ccw"])
    cy = sum(pt["y"] for pt in obj["polygon_ccw"]) / len(obj["polygon_ccw"])
    ax.text(cx, cy, obj["object_type"], fontsize=7, ha="center", va="center")

ax.set_aspect("equal", adjustable="box")
ax.invert_yaxis()
plt.show()
