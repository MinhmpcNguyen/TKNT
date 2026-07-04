import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from stylist.rules import get_soft_rule_templates

# -----------------------------
# SIZE TIERING (S/M/L)
# -----------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        raise ValueError("_percentile() on empty list")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


def _filter_outliers_iqr_logA(areas: list[float], k: float = 1.5) -> list[float]:
    if len(areas) < 5:
        return areas[:]
    log_as = [_safe_log(a) for a in areas]
    q1 = _percentile(log_as, 25)
    q3 = _percentile(log_as, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    kept = [a for a in areas if lo <= _safe_log(a) <= hi]
    return kept if kept else areas[:]


def _tier_of_area(area: float, a_low: float, a_high: float) -> str:
    if area <= a_low:
        return "S"
    if area <= a_high:
        return "M"
    return "L"


def _get_category(item: dict[str, Any]) -> str:
    attrs = item.get("attributes") or {}
    cat = attrs.get("category") or item.get("type") or "unknown"
    return str(cat)


def _compute_footprint_area_m2(item: dict[str, Any]) -> float | None:
    try:
        length_mm = float(item["length_mm"])
        width_mm = float(item["width_mm"])
    except Exception:
        return None
    if length_mm <= 0 or width_mm <= 0:
        return None
    l_m = length_mm / 1000.0
    w_m = width_mm / 1000.0
    if w_m > l_m:
        l_m, w_m = w_m, l_m
    return l_m * w_m


def annotate_inventory_size_tiers(
    inventory: list[dict[str, Any]],
    *,
    p_low: float = 30,
    p_high: float = 70,
    min_n_for_category_thresholds: int = 30,
    outlier_k: float = 1.5,
) -> None:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    area_by_item: dict[str, float] = {}

    for item in inventory:
        area = _compute_footprint_area_m2(item)
        if area is None or area <= 0:
            continue
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        area_by_item[item_id] = area
        cat = _get_category(item)
        by_cat.setdefault(cat, []).append(item)

    global_areas = list(area_by_item.values())
    filtered_global = _filter_outliers_iqr_logA(global_areas, k=outlier_k)

    for cat, items in by_cat.items():
        areas = [
            area_by_item[str(it.get("id", ""))]
            for it in items
            if str(it.get("id", "")) in area_by_item
        ]
        filtered_cat = _filter_outliers_iqr_logA(areas, k=outlier_k)
        base_areas = filtered_cat
        if len(base_areas) < min_n_for_category_thresholds:
            base_areas = filtered_global
        if not base_areas:
            continue
        a_low = _percentile(base_areas, p_low)
        a_high = _percentile(base_areas, p_high)
        if a_high < a_low:
            a_low, a_high = a_high, a_low

        for item in items:
            item_id = str(item.get("id", ""))
            area = area_by_item.get(item_id)
            if area is None:
                continue
            item["footprint_area_m2"] = area
            item["size_tier"] = _tier_of_area(area, a_low, a_high)


# -----------------------------
# CONFIG
# -----------------------------
SEED = 2026
CURRENCY = "VND"

ROOM_TYPES = [
    "bedroom",
    "living_room",
    "studio",
    "office",
    "kitchen",
]
STYLES = [
    "modern",
    "japandi",
    "scandinavian",
    "minimal",
    "industrial",
    "classic",
    "boho",
    "midcentury",
    "coastal",
    "wabi_sabi",
]
BRANDS = ["IKEA", "Hòa Phát", "Nội Thất X", "Nội Thất Y", "LocalBrand"]
MATERIALS = [
    "wood",
    "metal",
    "glass",
    "fabric",
    "leather",
    "engineered_wood",
    "rattan",
    "stone",
]

STYLE_PALETTES: dict[str, list[str]] = {
    "modern": ["#E6E1D6", "#B9C1C7", "#2F3A3F", "#C99C6B", "#8A8D8F"],
    "japandi": ["#E8E1D6", "#D1C7B8", "#9C8F7A", "#2E2A27", "#B27F5C"],
    "scandinavian": ["#F2F0EA", "#DAD7CF", "#B9C4C2", "#7A8C86", "#C9A37A"],
    "minimal": ["#F5F5F2", "#D9D9D6", "#B0B0A8", "#2C2C2C", "#A58F7A"],
    "industrial": ["#2E2E2E", "#4D4D4D", "#A67C52", "#B4B4B4", "#7E6D5A"],
    "classic": ["#F0E6D6", "#C7B299", "#8A6E4B", "#4F3B2A", "#BFA06A"],
    "boho": ["#EAD9C8", "#C96B4B", "#8E8D6B", "#5E7B6E", "#D8B273"],
    "midcentury": ["#E4D9C5", "#C77C47", "#4F6D7A", "#2F2F2F", "#B3A27A"],
    "coastal": ["#F6F5F0", "#BFD7E2", "#8EB1C7", "#50708C", "#D1B48C"],
    "wabi_sabi": ["#E4DED2", "#BFAF9A", "#8B7E6A", "#3F3A36", "#A38D77"],
}

STYLE_MATERIAL_HINTS: dict[str, list[str]] = {
    "japandi": ["light wood", "linen", "matte black"],
    "scandinavian": ["light wood", "wool", "white oak"],
    "minimal": ["matte", "smooth", "neutral"],
    "industrial": ["black metal", "aged wood", "concrete"],
    "classic": ["dark wood", "brass", "velvet"],
    "boho": ["rattan", "woven", "textured fabric"],
    "midcentury": ["walnut", "brass", "leather"],
    "coastal": ["washed wood", "linen", "light blue"],
    "wabi_sabi": ["raw wood", "stone", "linen"],
    "modern": ["glass", "metal", "neutral fabric"],
}

STYLE_LIGHTING_HINTS: dict[str, str] = {
    "japandi": "warm 2700-3000K, soft ambient layers",
    "scandinavian": "bright 3000-3500K, maximize daylight",
    "minimal": "neutral 3000K, reduce glare",
    "industrial": "warm 2700K with metal fixtures",
    "classic": "warm 2700K with layered accents",
    "boho": "warm 2700K with textured shades",
    "midcentury": "warm 2700-3000K with brass accents",
    "coastal": "neutral 3000K with airy diffusion",
    "wabi_sabi": "warm 2700K, low contrast",
    "modern": "neutral 3000-3500K, clean fixtures",
}
# ============================================================
# STYLE COLOR SYSTEMS (surface + object-role palettes)
# ============================================================

STYLE_COLOR_SYSTEMS: dict[str, dict[str, list[str]]] = {
    "modern": {
        "walls_main": ["#F4F1EA", "#ECE7DE", "#E3DDD2"],
        "walls_accent": ["#D6D0C6", "#CFC8BC"],
        "ceiling": ["#FAFAF7"],
        "floor_wood": ["#C9A47E", "#B88F68", "#7D5E47"],
        "floor_stone": ["#BDB7AE", "#A9A39A"],
        "large_casegoods": ["#F2EEE8", "#D8D2C8", "#7A5A43", "#3C3C3C"],
        "small_casegoods": ["#F2EEE8", "#D8D2C8", "#A68563", "#2F2F2F"],
        "upholstery": ["#E8E1D8", "#CFC6BB", "#AFA79C", "#7C858D"],
        "textiles": ["#F3EFE8", "#DDD4C8", "#C9C2B7", "#B7B09F"],
        "lighting": ["#111111", "#5B5B5B", "#B8B8B8"],
        "appliance": ["#111111", "#D9D9D9", "#A6A6A6"],
        "decor": ["#D2C2AF", "#B98D6E", "#8C9386", "#6E7B86"],
        "accent_small": ["#B97A56", "#7E8C7A", "#5F7487"],
        "neutral_misc": ["#E8E1D8", "#D2CBC1", "#B9B2A8"],
    },
    "japandi": {
        "walls_main": ["#EEE7DD", "#E5DDD1", "#DCCFBE"],
        "walls_accent": ["#C8B9A5", "#B9A58C"],
        "ceiling": ["#F7F3EC"],
        "floor_wood": ["#C79E72", "#AE845C", "#7B6046"],
        "floor_stone": ["#B7B0A4", "#A39A8D"],
        "large_casegoods": ["#E9E0D3", "#D7C9B8", "#9B7F63", "#3C342E"],
        "small_casegoods": ["#E9E0D3", "#CDBCA7", "#A58263", "#2F2A27"],
        "upholstery": ["#E7DED2", "#D5C7B6", "#B9AA98", "#8D7B67"],
        "textiles": ["#F1EADF", "#DDD1C0", "#C8B49A", "#A38C76"],
        "lighting": ["#2C2622", "#6F6257", "#B5A79B"],
        "appliance": ["#2E2A27", "#D8D2C9", "#B4A99A"],
        "decor": ["#CDB59B", "#A88367", "#8A8D74", "#6B625B"],
        "accent_small": ["#B27F5C", "#7E8A72", "#8E7760"],
        "neutral_misc": ["#E6DED2", "#CFC3B3", "#B6A694"],
    },
    "scandinavian": {
        "walls_main": ["#F6F4EE", "#F0EEE7", "#E8E4DB"],
        "walls_accent": ["#D9D5CC", "#CFCABF"],
        "ceiling": ["#FBFBF8"],
        "floor_wood": ["#D6B48C", "#C89E73", "#A97B56"],
        "floor_stone": ["#C4C1B8", "#B0ADA4"],
        "large_casegoods": ["#F7F4EE", "#E6E1D7", "#CBAE8B", "#6D746F"],
        "small_casegoods": ["#F7F4EE", "#DDD8CE", "#BDA07F", "#7A8781"],
        "upholstery": ["#E9E6DE", "#D9D6CF", "#C1C8C3", "#9AA7A2"],
        "textiles": ["#F2EFE8", "#E1DDD4", "#CBD4D0", "#B9C1BC"],
        "lighting": ["#222222", "#7E8683", "#CFCFCB"],
        "appliance": ["#E5E5E1", "#BFC6C3", "#1F1F1F"],
        "decor": ["#D0B08C", "#98ABA8", "#889B95", "#9E8E78"],
        "accent_small": ["#C8A27A", "#7E9A96", "#A1B6B0"],
        "neutral_misc": ["#F1EEE8", "#DDD9D0", "#C5C9C4"],
    },
    "minimal": {
        "walls_main": ["#F7F6F2", "#F0EEE8", "#E4E1D8"],
        "walls_accent": ["#D6D2C8", "#C9C5BB"],
        "ceiling": ["#FCFCFA"],
        "floor_wood": ["#C8A67C", "#B48F66", "#7A624C"],
        "floor_stone": ["#C0BBB2", "#AAA59D"],
        "large_casegoods": ["#F4F1EA", "#DEDBD3", "#A58E79", "#2D2D2D"],
        "small_casegoods": ["#F4F1EA", "#D7D3CB", "#B29A84", "#3B3B3B"],
        "upholstery": ["#ECE7DF", "#D5D0C8", "#B7B0A6", "#90928F"],
        "textiles": ["#F3EFE9", "#E0DBD2", "#CFC8BD", "#B8B0A5"],
        "lighting": ["#1D1D1D", "#666666", "#C8C8C5"],
        "appliance": ["#F0F0EC", "#A7A7A2", "#1D1D1D"],
        "decor": ["#C8B29B", "#9A9386", "#7B7F7C"],
        "accent_small": ["#A98E77", "#7D847D", "#8F9499"],
        "neutral_misc": ["#ECE7DF", "#D7D1C8", "#BBB4AA"],
    },
    "industrial": {
        "walls_main": ["#E3DED5", "#D4CEC4", "#C4BCB1"],
        "walls_accent": ["#64615B", "#4B4A46"],
        "ceiling": ["#F0EEE9"],
        "floor_wood": ["#9A7455", "#7F5F46", "#5F4737"],
        "floor_stone": ["#8F8B84", "#77736E"],
        "large_casegoods": ["#6B655D", "#A08870", "#3B3B39", "#B7AA98"],
        "small_casegoods": ["#7B746B", "#B59B81", "#222222", "#9D9488"],
        "upholstery": ["#7B7369", "#9B8C7B", "#545E64", "#B9ADA0"],
        "textiles": ["#C9BBAE", "#A99B8E", "#7B756E", "#585752"],
        "lighting": ["#111111", "#3E3E3E", "#8C8C8A"],
        "appliance": ["#111111", "#6E6E6E", "#B7B7B4"],
        "decor": ["#A67C52", "#6E7A74", "#8A6B58"],
        "accent_small": ["#A56E43", "#6D7A71", "#7B8890"],
        "neutral_misc": ["#BDB3A7", "#8F867B", "#5A5852"],
    },
    "classic": {
        "walls_main": ["#F2E9DA", "#E9DDCC", "#DED0BB"],
        "walls_accent": ["#C7B299", "#B79B7A"],
        "ceiling": ["#FBF7F0"],
        "floor_wood": ["#A57952", "#855E3E", "#65452D"],
        "floor_stone": ["#C8C0B5", "#B2A99E"],
        "large_casegoods": ["#DCC9AF", "#B89772", "#6D4D35", "#3E2F26"],
        "small_casegoods": ["#E4D5C2", "#C5A787", "#7B5B44", "#4A3930"],
        "upholstery": ["#D9CABB", "#C7B299", "#A98C70", "#7E5E4A"],
        "textiles": ["#F0E5D6", "#D9C5AE", "#C8B299", "#A9896A"],
        "lighting": ["#8C6A43", "#B08D57", "#3F3127"],
        "appliance": ["#D8D1C7", "#A49A8D", "#3F3127"],
        "decor": ["#B88A55", "#7F5E45", "#A9805B"],
        "accent_small": ["#A66B3D", "#8C5C47", "#B08D57"],
        "neutral_misc": ["#E7D9C8", "#CDB69E", "#A68F78"],
    },
    "boho": {
        "walls_main": ["#F3E7D8", "#E9D9C8", "#E0CEB8"],
        "walls_accent": ["#C98B67", "#B96E53"],
        "ceiling": ["#FAF4EC"],
        "floor_wood": ["#C28E63", "#A77249", "#7C573C"],
        "floor_stone": ["#C7B7A6", "#B6A08D"],
        "large_casegoods": ["#DAB997", "#B98A61", "#7B5A43", "#A57C67"],
        "small_casegoods": ["#E1C8A9", "#C79166", "#8C674A", "#6A5244"],
        "upholstery": ["#D8B39A", "#C96B4B", "#8E8D6B", "#A68573"],
        "textiles": ["#E8D2BE", "#D8B273", "#B96B4B", "#8E8D6B"],
        "lighting": ["#6B503F", "#A57C61", "#C7A782"],
        "appliance": ["#E0D5C9", "#A89B8E", "#675E58"],
        "decor": ["#C96B4B", "#D8B273", "#5E7B6E", "#9E6C5A"],
        "accent_small": ["#C96B4B", "#D8B273", "#5E7B6E"],
        "neutral_misc": ["#E4D1BF", "#C9B09A", "#A58A74"],
    },
    "midcentury": {
        "walls_main": ["#F1E8D9", "#E7DCCB", "#DCCFB8"],
        "walls_accent": ["#C2A47C", "#B58C60"],
        "ceiling": ["#FBF8F2"],
        "floor_wood": ["#B17B4F", "#8E603D", "#6F492F"],
        "floor_stone": ["#BFB6AA", "#AAA094"],
        "large_casegoods": ["#D7C1A5", "#A66C43", "#6A4A35", "#364B57"],
        "small_casegoods": ["#E2D0B8", "#C77C47", "#7A5A45", "#4F6D7A"],
        "upholstery": ["#DCC8AF", "#B69C7F", "#5F7480", "#768C7C"],
        "textiles": ["#E8D9C5", "#C9A37A", "#8FA09A", "#B49A7A"],
        "lighting": ["#8A6A44", "#C19A62", "#2F2F2F"],
        "appliance": ["#DED8CF", "#A9A195", "#30353A"],
        "decor": ["#C77C47", "#4F6D7A", "#879975", "#A87B56"],
        "accent_small": ["#C77C47", "#4F6D7A", "#7D946F"],
        "neutral_misc": ["#E2D6C8", "#C6B4A0", "#9F8E7B"],
    },
    "coastal": {
        "walls_main": ["#F8F6F1", "#EEF3F3", "#E3ECEB"],
        "walls_accent": ["#C8D9DF", "#B8CDD5"],
        "ceiling": ["#FCFCFA"],
        "floor_wood": ["#D5B692", "#C9A37C", "#A67F61"],
        "floor_stone": ["#C7CFD0", "#B0B9BC"],
        "large_casegoods": ["#F1EFE8", "#DCE7E6", "#9EB7C0", "#7A695A"],
        "small_casegoods": ["#F5F3ED", "#D6E0E0", "#8EB1C7", "#8E7763"],
        "upholstery": ["#E9E8E0", "#D7E3E4", "#BFD7E2", "#C9B8A0"],
        "textiles": ["#F6F5F0", "#DCE7E6", "#BFD7E2", "#D9C5AE"],
        "lighting": ["#6E7F86", "#B0B9BC", "#D7D7D3"],
        "appliance": ["#F1F1ED", "#B7C5C8", "#5E707A"],
        "decor": ["#8EB1C7", "#D1B48C", "#7FA0AE", "#A7BDB5"],
        "accent_small": ["#8EB1C7", "#50708C", "#D1B48C"],
        "neutral_misc": ["#ECEAE3", "#D5DFDE", "#BACACA"],
    },
    "wabi_sabi": {
        "walls_main": ["#E9E2D6", "#DED4C6", "#D2C5B2"],
        "walls_accent": ["#B9AA95", "#A38D77"],
        "ceiling": ["#F6F1EA"],
        "floor_wood": ["#BA916A", "#9B7556", "#74563F"],
        "floor_stone": ["#B8B0A4", "#A39A8F"],
        "large_casegoods": ["#D8CBB8", "#BFAF9A", "#8B7E6A", "#4B433C"],
        "small_casegoods": ["#E0D4C3", "#C7B5A0", "#9C8A74", "#5E544B"],
        "upholstery": ["#D7CCBE", "#C1B2A0", "#A38D77", "#7B7166"],
        "textiles": ["#E7DED3", "#D3C6B6", "#B8A896", "#9B8B7A"],
        "lighting": ["#4A433D", "#877C73", "#B9AFA5"],
        "appliance": ["#DED8CF", "#AFA79C", "#5C554F"],
        "decor": ["#A38D77", "#8B7E6A", "#746A60", "#B59C84"],
        "accent_small": ["#A38D77", "#7B7166", "#8E7F70"],
        "neutral_misc": ["#DCCFC0", "#C2B2A0", "#A0907E"],
    },
}


def _palette_text(colors: list[str]) -> str:
    return ", ".join(colors)


def _infer_color_role(type_name: str) -> str:
    t = type_name.lower()

    if t in {"curtain", "blind", "rug", "cushion", "throw_blanket"}:
        return "textiles"

    if "lamp" in t or "light" in t or t in {"wall_sconce", "track_light"}:
        return "lighting"

    if t in {
        "tv",
        "monitor",
        "laptop",
        "desktop_pc",
        "printer",
        "microwave",
        "rice_cooker",
        "electric_kettle",
        "coffee_machine",
        "toaster",
        "air_fryer",
        "blender",
        "projector",
        "speaker",
        "smart_speaker",
        "fridge",
        "stove",
        "oven",
        "range_hood",
        "cooktop",
        "dishwasher",
        "washing_machine",
        "dryer",
        "air_conditioner",
        "fan",
        "heater",
        "dehumidifier",
        "air_purifier",
    }:
        return "appliance"

    if t in {
        "sofa",
        "sectional_sofa",
        "armchair",
        "recliner",
        "bean_bag",
        "ottoman",
        "pet_bed",
    }:
        return "upholstery"

    if t in {
        "chair",
        "dining_chair",
        "office_chair",
        "bar_stool",
        "bench",
        "entry_bench",
        "bathroom_stool",
    }:
        return "upholstery"

    if t in {"decor", "vase", "plant", "mirror", "wall_art", "clock"}:
        return "decor"

    if t in {"bedside_lamp", "table_lamp", "desk_lamp", "floor_lamp"}:
        return "lighting"

    if any(
        k in t
        for k in [
            "wardrobe",
            "dresser",
            "nightstand",
            "desk",
            "bookshelf",
            "cabinet",
            "console",
            "table",
            "rack",
            "sideboard",
            "vanity",
            "shelf",
            "island",
            "bed",
            "headboard",
            "shoe_rack",
            "coat_rack",
            "umbrella_stand",
            "media_shelf",
            "filing_cabinet",
            "office_pedestal",
            "buffet_sideboard",
            "china_cabinet",
            "pantry_cabinet",
            "wine_cabinet",
            "bar_cart",
            "medicine_cabinet",
            "bathroom_shelf",
            "utility_sink",
            "ironing_board",
            "drying_rack",
            "projector_screen",
        ]
    ):
        # chia large/small cho text đẹp hơn
        if t in {
            "wardrobe",
            "dresser",
            "desk",
            "bookshelf",
            "storage_cabinet",
            "tv_console",
            "console_table",
            "dining_table",
            "kitchen_base_cabinet",
            "kitchen_wall_cabinet",
            "kitchen_tall_cabinet",
            "kitchen_island",
            "buffet_sideboard",
            "china_cabinet",
            "bathroom_vanity",
            "pantry_cabinet",
            "wine_cabinet",
            "filing_cabinet",
        }:
            return "large_casegoods"
        return "small_casegoods"

    return "neutral_misc"


def build_style_color_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    role_explainer = {
        "large_casegoods": "large storage and main hard furniture such as wardrobe, dresser, desk, cabinet, bookshelf, dining table",
        "small_casegoods": "small hard furniture such as nightstand, side table, coffee table, entry bench, compact support pieces",
        "upholstery": "sofa, armchair, chair, bench, soft seating, padded bed elements",
        "textiles": "curtain, blind, rug, cushion, throw blanket and other soft layers",
        "lighting": "floor lamp, table lamp, bedside lamp, pendant, ceiling and wall lighting hardware",
        "appliance": "TV, monitor, laptop, fridge, stove, microwave and visible electronics",
        "decor": "mirror, wall art, vase, plant, small decorative objects",
        "neutral_misc": "miscellaneous objects that should stay visually quiet",
    }

    for style in STYLES:
        system = STYLE_COLOR_SYSTEMS.get(style, STYLE_COLOR_SYSTEMS["modern"])

        # ---- room surfaces ----
        docs.append(
            {
                "doc_id": f"style_surfaces_{style}",
                "title": f"{style} surfaces: wall, floor, ceiling",
                "tags": [
                    "style",
                    "styling",
                    style,
                    "color",
                    "wall",
                    "floor",
                    "ceiling",
                    "surfaces",
                ],
                "text": (
                    f"For {style} interiors, recommended wall base colors: {_palette_text(system['walls_main'])}. "
                    f"Accent wall colors (use sparingly): {_palette_text(system['walls_accent'])}. "
                    f"Ceiling colors: {_palette_text(system['ceiling'])}. "
                    f"Recommended wood floor tones: {_palette_text(system['floor_wood'])}. "
                    f"Recommended stone/concrete floor tones: {_palette_text(system['floor_stone'])}. "
                    f"Rule: walls and floor must stay calmer than the main furniture anchor. "
                    f"Do not use strong saturated wall colors for {style} unless the room is explicitly designed around a bold feature wall."
                ),
            }
        )

        # ---- whole-room color strategy ----
        docs.append(
            {
                "doc_id": f"style_color_strategy_{style}",
                "title": f"{style} color strategy by role",
                "tags": ["style", "styling", style, "color", "strategy", "role_based"],
                "text": (
                    f"In {style}, use large_casegoods colors from {_palette_text(system['large_casegoods'])}. "
                    f"Use small_casegoods colors from {_palette_text(system['small_casegoods'])}. "
                    f"Use upholstery colors from {_palette_text(system['upholstery'])}. "
                    f"Use textile colors from {_palette_text(system['textiles'])}. "
                    f"Use lighting hardware colors from {_palette_text(system['lighting'])}. "
                    f"Use appliance visible colors from {_palette_text(system['appliance'])}. "
                    f"Use decor colors from {_palette_text(system['decor'])}. "
                    f"Accent colors must come mainly from {_palette_text(system['accent_small'])} and should stay limited to small objects or soft accessories. "
                    f"Rule: large furniture should stay neutral or wood-based; accent colors must not dominate wardrobes, desks, beds, sofas, or full walls."
                ),
            }
        )

        # ---- role docs ----
        for role, expl in role_explainer.items():
            if role not in system:
                continue
            docs.append(
                {
                    "doc_id": f"style_role_{style}_{role}",
                    "title": f"{style} color guidance for {role}",
                    "tags": ["style", "styling", style, "color", role],
                    "text": (
                        f"For {style}, {expl} should prefer colors: {_palette_text(system[role])}. "
                        f"Keep these pieces coordinated with wall colors {_palette_text(system['walls_main'])} "
                        f"and floor tones {_palette_text(system['floor_wood'])} or {_palette_text(system['floor_stone'])}. "
                        f"Avoid using accent_small colors {_palette_text(system['accent_small'])} as the dominant finish for {role}."
                    ),
                }
            )

        # ---- per furniture type docs ----
        for type_name in CATEGORY_SPECS.keys():
            role = _infer_color_role(type_name)
            palette = system.get(role) or system["neutral_misc"]

            docs.append(
                {
                    "doc_id": f"style_item_{style}_{type_name}",
                    "title": f"{style} color guidance for {type_name}",
                    "tags": [
                        "style",
                        "styling",
                        style,
                        "color",
                        "item",
                        type_name,
                        role,
                    ],
                    "text": (
                        f"For {style}, recommended colors for {type_name} are: {_palette_text(palette)}. "
                        f"This item belongs to role {role}. "
                        f"Keep it coordinated with wall colors {_palette_text(system['walls_main'])} "
                        f"and floor tones {_palette_text(system['floor_wood']) if role != 'appliance' else _palette_text(system['floor_stone'])}. "
                        f"If {type_name} is visually large, avoid strong accent_small colors {_palette_text(system['accent_small'])} as the primary finish. "
                        f"If {type_name} is small decor or accessory, one accent_small color may be used, but only as a minor accent."
                    ),
                }
            )

    return docs


# type/category -> size ranges (cm, converted to mm in output) + price ranges (VND)
CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    # structural
    "door": {
        "count": 6,
        "length_cm": (8, 12),
        "width_cm": (80, 110),
        "height_cm": (200, 230),
        "price_amount": (1500000, 6000000),
        "attributes": lambda rnd: {
            "open_angle_deg": rnd.choice([80, 90, 100]),
            "default_open_direction": "inward",
            "hinge_side": rnd.choice(["left", "right"]),
        },
    },
    "window": {
        "count": 10,
        "length_cm": (8, 14),
        "width_cm": (90, 240),
        "height_cm": (80, 180),
        "price_amount": (800000, 5500000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "frame": rnd.choice(["aluminum", "wood"]),
        },
    },
    # furniture
    "bed": {
        "count": 14,
        "length_cm": (180, 220),
        "width_cm": (120, 200),
        "height_cm": (30, 65),
        "price_amount": (4000000, 18000000),
        "attributes": lambda rnd: {"headboard": rnd.choice([True, False])},
    },
    "wardrobe": {
        "count": 14,
        "length_cm": (45, 65),
        "width_cm": (80, 260),
        "height_cm": (180, 240),
        "price_amount": (3500000, 24000000),
        "attributes": lambda rnd: {
            "doors": rnd.choice([2, 3, 4]),
            "sliding": rnd.choice([True, False]),
        },
    },
    "dresser": {
        "count": 10,
        "length_cm": (40, 55),
        "width_cm": (80, 140),
        "height_cm": (70, 100),
        "price_amount": (1800000, 12000000),
        "attributes": lambda rnd: {"drawers": rnd.choice([3, 4, 6])},
    },
    "nightstand": {
        "count": 10,
        "length_cm": (35, 45),
        "width_cm": (35, 55),
        "height_cm": (40, 65),
        "price_amount": (500000, 3000000),
        "attributes": lambda rnd: {"drawers": rnd.choice([1, 2])},
    },
    "bench": {
        "count": 8,
        "length_cm": (40, 55),
        "width_cm": (90, 140),
        "height_cm": (40, 50),
        "price_amount": (900000, 6000000),
        "attributes": lambda rnd: {"upholstered": rnd.choice([True, False])},
    },
    "desk": {
        "count": 14,
        "length_cm": (50, 75),
        "width_cm": (90, 180),
        "height_cm": (70, 78),
        "price_amount": (1200000, 9000000),
        "attributes": lambda rnd: {"drawers": rnd.choice([0, 1, 2])},
    },
    "chair": {
        "count": 18,
        "length_cm": (45, 60),
        "width_cm": (45, 60),
        "height_cm": (75, 105),
        "price_amount": (400000, 3500000),
        "attributes": lambda rnd: {"with_armrest": rnd.choice([True, False])},
    },
    "armchair": {
        "count": 10,
        "length_cm": (70, 90),
        "width_cm": (70, 95),
        "height_cm": (70, 95),
        "price_amount": (2500000, 14000000),
        "attributes": lambda rnd: {"seat_depth": rnd.choice([55, 60, 65])},
    },
    "sofa": {
        "count": 14,
        "length_cm": (80, 110),
        "width_cm": (160, 300),
        "height_cm": (70, 95),
        "price_amount": (6000000, 45000000),
        "attributes": lambda rnd: {
            "seats": rnd.choice([2, 3, 4]),
            "corner": rnd.choice([True, False]),
        },
    },
    "ottoman": {
        "count": 8,
        "length_cm": (45, 65),
        "width_cm": (45, 85),
        "height_cm": (35, 45),
        "price_amount": (800000, 6000000),
        "attributes": lambda rnd: {"shape": rnd.choice(["rect", "round"])},
    },
    "coffee_table": {
        "count": 12,
        "length_cm": (45, 70),
        "width_cm": (80, 150),
        "height_cm": (30, 45),
        "price_amount": (800000, 7000000),
        "attributes": lambda rnd: {"shape": rnd.choice(["rect", "oval", "round"])},
    },
    "side_table": {
        "count": 12,
        "length_cm": (35, 50),
        "width_cm": (35, 50),
        "height_cm": (45, 60),
        "price_amount": (500000, 3500000),
        "attributes": lambda rnd: {"shape": rnd.choice(["round", "square"])},
    },
    "console_table": {
        "count": 8,
        "length_cm": (30, 45),
        "width_cm": (100, 160),
        "height_cm": (70, 85),
        "price_amount": (1200000, 9000000),
        "attributes": lambda rnd: {"storage": rnd.choice([True, False])},
    },
    "tv_console": {
        "count": 12,
        "length_cm": (35, 50),
        "width_cm": (120, 240),
        "height_cm": (35, 60),
        "price_amount": (1500000, 12000000),
        "attributes": lambda rnd: {"cable_management": True},
    },
    "tv": {
        "count": 8,
        "length_cm": (5, 8),
        "width_cm": (90, 180),
        "height_cm": (50, 100),
        "price_amount": (6000000, 25000000),
        "attributes": lambda rnd: {"wall_mountable": rnd.choice([True, False])},
    },
    "bookshelf": {
        "count": 12,
        "length_cm": (25, 40),
        "width_cm": (60, 140),
        "height_cm": (150, 240),
        "price_amount": (900000, 9000000),
        "attributes": lambda rnd: {"shelves": rnd.choice([4, 5, 6])},
    },
    "storage_cabinet": {
        "count": 10,
        "length_cm": (40, 55),
        "width_cm": (60, 120),
        "height_cm": (120, 200),
        "price_amount": (1500000, 12000000),
        "attributes": lambda rnd: {"doors": rnd.choice([2, 3])},
    },
    "dining_table": {
        "count": 12,
        "length_cm": (70, 95),
        "width_cm": (120, 220),
        "height_cm": (72, 76),
        "price_amount": (2500000, 22000000),
        "attributes": lambda rnd: {"seats": rnd.choice([4, 6, 8])},
    },
    "dining_chair": {
        "count": 18,
        "length_cm": (45, 55),
        "width_cm": (45, 55),
        "height_cm": (80, 100),
        "price_amount": (450000, 3500000),
        "attributes": lambda rnd: {"stackable": rnd.choice([True, False])},
    },
    "rug": {
        "count": 14,
        "length_cm": (120, 240),
        "width_cm": (160, 300),
        "height_cm": (1, 2),
        "price_amount": (900000, 8000000),
        "attributes": lambda rnd: {"shape": rnd.choice(["rect", "round"])},
    },
    "mirror": {
        "count": 10,
        "length_cm": (3, 6),
        "width_cm": (50, 120),
        "height_cm": (80, 180),
        "price_amount": (500000, 6000000),
        "attributes": lambda rnd: {"shape": rnd.choice(["rect", "round"])},
    },
    "plant": {
        "count": 12,
        "length_cm": (25, 45),
        "width_cm": (25, 45),
        "height_cm": (60, 180),
        "price_amount": (200000, 2500000),
        "attributes": lambda rnd: {"pot": rnd.choice(["ceramic", "woven", "metal"])},
    },
    "decor": {
        "count": 12,
        "length_cm": (15, 35),
        "width_cm": (15, 35),
        "height_cm": (15, 45),
        "price_amount": (150000, 1500000),
        "attributes": lambda rnd: {"type": rnd.choice(["vase", "frame", "sculpture"])},
    },
    "floor_lamp": {
        "count": 8,
        "length_cm": (30, 45),
        "width_cm": (30, 45),
        "height_cm": (140, 180),
        "price_amount": (800000, 5000000),
        "attributes": lambda rnd: {"mount": "floor"},
    },
    "desk_lamp": {
        "count": 8,
        "length_cm": (15, 25),
        "width_cm": (15, 25),
        "height_cm": (30, 55),
        "price_amount": (300000, 2500000),
        "attributes": lambda rnd: {"mount": "table"},
    },
    "pendant_light": {
        "count": 8,
        "length_cm": (20, 40),
        "width_cm": (20, 40),
        "height_cm": (25, 60),
        "price_amount": (600000, 4500000),
        "attributes": lambda rnd: {"mount": "ceiling"},
    },
    # kitchen
    "kitchen_base_cabinet": {
        "count": 14,
        "length_cm": (55, 65),
        "width_cm": (40, 120),
        "height_cm": (80, 92),
        "price_amount": (1200000, 8500000),
        "attributes": lambda rnd: {"module": True},
    },
    "kitchen_wall_cabinet": {
        "count": 12,
        "length_cm": (30, 40),
        "width_cm": (40, 120),
        "height_cm": (50, 90),
        "price_amount": (900000, 6500000),
        "attributes": lambda rnd: {"mount": "wall"},
    },
    "kitchen_tall_cabinet": {
        "count": 6,
        "length_cm": (55, 70),
        "width_cm": (50, 90),
        "height_cm": (200, 240),
        "price_amount": (2000000, 12000000),
        "attributes": lambda rnd: {"tall": True},
    },
    "kitchen_island": {
        "count": 6,
        "length_cm": (70, 100),
        "width_cm": (120, 220),
        "height_cm": (85, 95),
        "price_amount": (3500000, 22000000),
        "attributes": lambda rnd: {"seating": rnd.choice([2, 3])},
    },
    "fridge": {
        "count": 6,
        "length_cm": (60, 75),
        "width_cm": (55, 95),
        "height_cm": (160, 195),
        "price_amount": (6000000, 25000000),
        "attributes": lambda rnd: {
            "door_swing": "right" if rnd.random() < 0.5 else "left"
        },
    },
    "stove": {
        "count": 6,
        "length_cm": (55, 70),
        "width_cm": (55, 70),
        "height_cm": (85, 95),
        "price_amount": (2500000, 12000000),
        "attributes": lambda rnd: {"fuel": rnd.choice(["gas", "electric"])},
    },
    "sink": {
        "count": 6,
        "length_cm": (50, 70),
        "width_cm": (45, 60),
        "height_cm": (85, 95),
        "price_amount": (1200000, 7000000),
        "attributes": lambda rnd: {"basin": rnd.choice([1, 2])},
    },
    "microwave": {
        "count": 6,
        "length_cm": (35, 45),
        "width_cm": (45, 60),
        "height_cm": (25, 35),
        "price_amount": (1500000, 6000000),
        "attributes": lambda rnd: {"countertop": True},
    },
    # bathroom
    "bathroom_vanity": {
        "count": 6,
        "length_cm": (50, 70),
        "width_cm": (80, 140),
        "height_cm": (80, 90),
        "price_amount": (2500000, 15000000),
        "attributes": lambda rnd: {"basin": rnd.choice([1, 2])},
    },
    "toilet": {
        "count": 6,
        "length_cm": (55, 70),
        "width_cm": (35, 45),
        "height_cm": (70, 80),
        "price_amount": (1800000, 9000000),
        "attributes": lambda rnd: {"flush": rnd.choice(["dual", "single"])},
    },
    "shower": {
        "count": 4,
        "length_cm": (80, 120),
        "width_cm": (80, 120),
        "height_cm": (190, 220),
        "price_amount": (5000000, 25000000),
        "attributes": lambda rnd: {"enclosure": rnd.choice(["glass", "tile"])},
    },
    "bathtub": {
        "count": 4,
        "length_cm": (150, 180),
        "width_cm": (70, 90),
        "height_cm": (55, 65),
        "price_amount": (7000000, 28000000),
        "attributes": lambda rnd: {"type": rnd.choice(["freestanding", "built_in"])},
    },
    "washing_machine": {
        "count": 4,
        "length_cm": (55, 70),
        "width_cm": (55, 70),
        "height_cm": (80, 95),
        "price_amount": (5000000, 20000000),
        "attributes": lambda rnd: {"load": rnd.choice(["front", "top"])},
    },
    "dryer": {
        "count": 4,
        "length_cm": (55, 70),
        "width_cm": (55, 70),
        "height_cm": (80, 95),
        "price_amount": (5000000, 20000000),
        "attributes": lambda rnd: {"heat": rnd.choice(["vented", "heat_pump"])},
    },
    # =========================
    # bedroom / soft furnishing
    # =========================
    "curtain": {
        "count": 12,
        "length_cm": (1, 3),
        "width_cm": (120, 320),
        "height_cm": (180, 300),
        "price_amount": (300000, 5000000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "blackout": rnd.choice([True, False]),
            "open_style": rnd.choice(["center_split", "one_side"]),
        },
    },
    "blind": {
        "count": 8,
        "length_cm": (2, 6),
        "width_cm": (60, 220),
        "height_cm": (80, 240),
        "price_amount": (400000, 4500000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "type": rnd.choice(["roller", "roman", "venetian"]),
        },
    },
    "mattress": {
        "count": 10,
        "length_cm": (18, 35),
        "width_cm": (100, 200),
        "height_cm": (190, 220),
        "price_amount": (2500000, 18000000),
        "attributes": lambda rnd: {
            "firmness": rnd.choice(["soft", "medium", "firm"]),
        },
    },
    "headboard": {
        "count": 8,
        "length_cm": (6, 12),
        "width_cm": (100, 220),
        "height_cm": (90, 140),
        "price_amount": (1200000, 9000000),
        "attributes": lambda rnd: {
            "upholstered": rnd.choice([True, False]),
            "mount": rnd.choice(["bed", "wall"]),
        },
    },
    "bedside_lamp": {
        "count": 10,
        "length_cm": (18, 30),
        "width_cm": (18, 30),
        "height_cm": (30, 55),
        "price_amount": (250000, 2500000),
        "attributes": lambda rnd: {
            "mount": "table",
            "shade": rnd.choice(["fabric", "glass", "metal"]),
        },
    },
    "wall_art": {
        "count": 14,
        "length_cm": (2, 5),
        "width_cm": (40, 140),
        "height_cm": (40, 100),
        "price_amount": (300000, 5000000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "type": rnd.choice(["painting", "print", "poster"]),
        },
    },
    "laundry_basket": {
        "count": 6,
        "length_cm": (30, 45),
        "width_cm": (30, 45),
        "height_cm": (45, 65),
        "price_amount": (150000, 1200000),
        "attributes": lambda rnd: {
            "material": rnd.choice(["plastic", "woven", "fabric"]),
        },
    },
    # =========================
    # office / workstation
    # =========================
    "office_chair": {
        "count": 12,
        "length_cm": (55, 75),
        "width_cm": (55, 75),
        "height_cm": (95, 125),
        "price_amount": (1200000, 12000000),
        "attributes": lambda rnd: {
            "wheels": True,
            "with_armrest": rnd.choice([True, False]),
        },
    },
    "filing_cabinet": {
        "count": 8,
        "length_cm": (45, 60),
        "width_cm": (40, 90),
        "height_cm": (60, 135),
        "price_amount": (1200000, 8000000),
        "attributes": lambda rnd: {
            "drawers": rnd.choice([2, 3, 4]),
            "lockable": rnd.choice([True, False]),
        },
    },
    "office_pedestal": {
        "count": 10,
        "length_cm": (45, 55),
        "width_cm": (35, 45),
        "height_cm": (55, 70),
        "price_amount": (800000, 4500000),
        "attributes": lambda rnd: {
            "drawers": rnd.choice([2, 3]),
            "mobile": rnd.choice([True, False]),
        },
    },
    "monitor": {
        "count": 12,
        "length_cm": (12, 25),
        "width_cm": (50, 90),
        "height_cm": (30, 60),
        "price_amount": (2500000, 18000000),
        "attributes": lambda rnd: {
            "wall_mountable": rnd.choice([True, False]),
            "size_inch": rnd.choice([24, 27, 32, 34]),
        },
    },
    "laptop": {
        "count": 12,
        "length_cm": (1, 3),
        "width_cm": (30, 40),
        "height_cm": (20, 30),
        "price_amount": (8000000, 45000000),
        "attributes": lambda rnd: {
            "screen_inch": rnd.choice([13, 14, 15, 16]),
            "portable": True,
        },
    },
    "desktop_pc": {
        "count": 8,
        "length_cm": (30, 55),
        "width_cm": (15, 28),
        "height_cm": (35, 55),
        "price_amount": (10000000, 50000000),
        "attributes": lambda rnd: {
            "form_factor": rnd.choice(["tower", "mini_pc"]),
        },
    },
    "keyboard": {
        "count": 10,
        "length_cm": (2, 5),
        "width_cm": (28, 48),
        "height_cm": (12, 18),
        "price_amount": (300000, 3500000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["fullsize", "tkl", "compact"]),
        },
    },
    "mouse": {
        "count": 10,
        "length_cm": (3, 5),
        "width_cm": (6, 8),
        "height_cm": (10, 14),
        "price_amount": (150000, 2500000),
        "attributes": lambda rnd: {
            "wireless": rnd.choice([True, False]),
        },
    },
    "printer": {
        "count": 6,
        "length_cm": (20, 35),
        "width_cm": (35, 55),
        "height_cm": (30, 50),
        "price_amount": (1500000, 12000000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["inkjet", "laser"]),
        },
    },
    "whiteboard": {
        "count": 6,
        "length_cm": (2, 5),
        "width_cm": (60, 180),
        "height_cm": (45, 120),
        "price_amount": (500000, 5000000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "magnetic": rnd.choice([True, False]),
        },
    },
    # =========================
    # living room / media
    # =========================
    "sectional_sofa": {
        "count": 8,
        "length_cm": (85, 110),
        "width_cm": (220, 360),
        "height_cm": (70, 95),
        "price_amount": (12000000, 60000000),
        "attributes": lambda rnd: {
            "seats": rnd.choice([3, 4, 5]),
            "chaise_side": rnd.choice(["left", "right"]),
        },
    },
    "recliner": {
        "count": 6,
        "length_cm": (85, 105),
        "width_cm": (75, 100),
        "height_cm": (95, 110),
        "price_amount": (4000000, 18000000),
        "attributes": lambda rnd: {
            "manual": rnd.choice([True, False]),
        },
    },
    "bean_bag": {
        "count": 6,
        "length_cm": (70, 110),
        "width_cm": (70, 110),
        "height_cm": (60, 90),
        "price_amount": (800000, 4000000),
        "attributes": lambda rnd: {
            "shape": rnd.choice(["round", "pear"]),
        },
    },
    "media_shelf": {
        "count": 6,
        "length_cm": (25, 40),
        "width_cm": (80, 180),
        "height_cm": (80, 180),
        "price_amount": (1200000, 9000000),
        "attributes": lambda rnd: {
            "open_shelves": rnd.choice([True, False]),
        },
    },
    "speaker": {
        "count": 8,
        "length_cm": (15, 35),
        "width_cm": (12, 30),
        "height_cm": (20, 100),
        "price_amount": (1000000, 25000000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["bookshelf", "floorstanding", "soundbar"]),
        },
    },
    "projector": {
        "count": 4,
        "length_cm": (18, 35),
        "width_cm": (25, 45),
        "height_cm": (8, 18),
        "price_amount": (6000000, 50000000),
        "attributes": lambda rnd: {
            "mount": rnd.choice(["table", "ceiling"]),
        },
    },
    "projector_screen": {
        "count": 4,
        "length_cm": (5, 12),
        "width_cm": (120, 300),
        "height_cm": (100, 220),
        "price_amount": (1500000, 12000000),
        "attributes": lambda rnd: {
            "mount": rnd.choice(["wall", "ceiling"]),
        },
    },
    # =========================
    # dining / bar / entry
    # =========================
    "bar_stool": {
        "count": 10,
        "length_cm": (40, 50),
        "width_cm": (40, 50),
        "height_cm": (65, 80),
        "price_amount": (600000, 4500000),
        "attributes": lambda rnd: {
            "backrest": rnd.choice([True, False]),
        },
    },
    "buffet_sideboard": {
        "count": 8,
        "length_cm": (40, 55),
        "width_cm": (120, 220),
        "height_cm": (75, 95),
        "price_amount": (2500000, 18000000),
        "attributes": lambda rnd: {
            "doors": rnd.choice([2, 3, 4]),
        },
    },
    "china_cabinet": {
        "count": 6,
        "length_cm": (35, 50),
        "width_cm": (80, 160),
        "height_cm": (180, 240),
        "price_amount": (4000000, 25000000),
        "attributes": lambda rnd: {
            "glass_doors": True,
        },
    },
    "entry_bench": {
        "count": 6,
        "length_cm": (35, 45),
        "width_cm": (80, 140),
        "height_cm": (40, 50),
        "price_amount": (900000, 6000000),
        "attributes": lambda rnd: {
            "storage": rnd.choice([True, False]),
        },
    },
    "shoe_rack": {
        "count": 8,
        "length_cm": (25, 40),
        "width_cm": (60, 120),
        "height_cm": (45, 120),
        "price_amount": (400000, 3500000),
        "attributes": lambda rnd: {
            "tiers": rnd.choice([2, 3, 4, 5]),
        },
    },
    "coat_rack": {
        "count": 6,
        "length_cm": (30, 45),
        "width_cm": (30, 45),
        "height_cm": (150, 185),
        "price_amount": (300000, 2500000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["standing", "wall"]),
        },
    },
    "umbrella_stand": {
        "count": 4,
        "length_cm": (20, 30),
        "width_cm": (20, 30),
        "height_cm": (40, 65),
        "price_amount": (150000, 1200000),
        "attributes": lambda rnd: {
            "material": rnd.choice(["metal", "ceramic", "plastic"]),
        },
    },
    # =========================
    # kitchen - major
    # =========================
    "dishwasher": {
        "count": 4,
        "length_cm": (55, 65),
        "width_cm": (55, 65),
        "height_cm": (80, 90),
        "price_amount": (7000000, 22000000),
        "attributes": lambda rnd: {
            "built_in": rnd.choice([True, False]),
        },
    },
    "oven": {
        "count": 6,
        "length_cm": (50, 65),
        "width_cm": (55, 65),
        "height_cm": (55, 70),
        "price_amount": (5000000, 25000000),
        "attributes": lambda rnd: {
            "built_in": rnd.choice([True, False]),
        },
    },
    "range_hood": {
        "count": 6,
        "length_cm": (35, 55),
        "width_cm": (60, 100),
        "height_cm": (20, 45),
        "price_amount": (2500000, 15000000),
        "attributes": lambda rnd: {
            "mount": rnd.choice(["wall", "ceiling", "under_cabinet"]),
        },
    },
    "cooktop": {
        "count": 6,
        "length_cm": (5, 12),
        "width_cm": (55, 90),
        "height_cm": (35, 55),
        "price_amount": (3000000, 18000000),
        "attributes": lambda rnd: {
            "fuel": rnd.choice(["gas", "electric", "induction"]),
        },
    },
    "pantry_cabinet": {
        "count": 6,
        "length_cm": (45, 65),
        "width_cm": (60, 120),
        "height_cm": (180, 240),
        "price_amount": (2500000, 18000000),
        "attributes": lambda rnd: {
            "pull_out": rnd.choice([True, False]),
        },
    },
    "wine_cabinet": {
        "count": 4,
        "length_cm": (45, 65),
        "width_cm": (30, 65),
        "height_cm": (80, 185),
        "price_amount": (5000000, 30000000),
        "attributes": lambda rnd: {
            "zone": rnd.choice([1, 2]),
        },
    },
    "bar_cart": {
        "count": 4,
        "length_cm": (35, 50),
        "width_cm": (60, 100),
        "height_cm": (75, 95),
        "price_amount": (800000, 7000000),
        "attributes": lambda rnd: {
            "wheels": True,
        },
    },
    # =========================
    # kitchen - small appliances
    # =========================
    "rice_cooker": {
        "count": 6,
        "length_cm": (20, 35),
        "width_cm": (20, 35),
        "height_cm": (20, 35),
        "price_amount": (500000, 5000000),
        "attributes": lambda rnd: {
            "capacity_l": rnd.choice([1.0, 1.8, 2.2]),
        },
    },
    "electric_kettle": {
        "count": 6,
        "length_cm": (18, 28),
        "width_cm": (18, 28),
        "height_cm": (20, 30),
        "price_amount": (250000, 2000000),
        "attributes": lambda rnd: {
            "capacity_l": rnd.choice([1.5, 1.7, 2.0]),
        },
    },
    "coffee_machine": {
        "count": 6,
        "length_cm": (22, 40),
        "width_cm": (18, 35),
        "height_cm": (28, 45),
        "price_amount": (1500000, 20000000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["drip", "espresso", "capsule"]),
        },
    },
    "toaster": {
        "count": 4,
        "length_cm": (18, 30),
        "width_cm": (25, 40),
        "height_cm": (15, 25),
        "price_amount": (300000, 2500000),
        "attributes": lambda rnd: {
            "slots": rnd.choice([2, 4]),
        },
    },
    "air_fryer": {
        "count": 4,
        "length_cm": (28, 40),
        "width_cm": (25, 38),
        "height_cm": (28, 40),
        "price_amount": (1200000, 5000000),
        "attributes": lambda rnd: {
            "capacity_l": rnd.choice([4, 6, 8]),
        },
    },
    "blender": {
        "count": 4,
        "length_cm": (15, 25),
        "width_cm": (15, 25),
        "height_cm": (30, 45),
        "price_amount": (500000, 3500000),
        "attributes": lambda rnd: {
            "jar": rnd.choice(["glass", "plastic"]),
        },
    },
    # =========================
    # bathroom
    # =========================
    "medicine_cabinet": {
        "count": 6,
        "length_cm": (12, 22),
        "width_cm": (40, 90),
        "height_cm": (50, 90),
        "price_amount": (700000, 5000000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "mirror_front": rnd.choice([True, False]),
        },
    },
    "towel_rack": {
        "count": 8,
        "length_cm": (8, 15),
        "width_cm": (40, 90),
        "height_cm": (8, 20),
        "price_amount": (150000, 1200000),
        "attributes": lambda rnd: {
            "mount": "wall",
        },
    },
    "shower_niche": {
        "count": 4,
        "length_cm": (8, 18),
        "width_cm": (30, 60),
        "height_cm": (30, 60),
        "price_amount": (300000, 2000000),
        "attributes": lambda rnd: {
            "mount": "wall",
        },
    },
    "bathroom_shelf": {
        "count": 6,
        "length_cm": (10, 20),
        "width_cm": (30, 80),
        "height_cm": (10, 25),
        "price_amount": (150000, 1800000),
        "attributes": lambda rnd: {
            "mount": rnd.choice(["wall", "corner"]),
        },
    },
    "bidet": {
        "count": 2,
        "length_cm": (55, 70),
        "width_cm": (35, 45),
        "height_cm": (38, 45),
        "price_amount": (2500000, 12000000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["standalone", "integrated"]),
        },
    },
    "bathroom_stool": {
        "count": 4,
        "length_cm": (25, 40),
        "width_cm": (25, 40),
        "height_cm": (25, 45),
        "price_amount": (150000, 1500000),
        "attributes": lambda rnd: {
            "waterproof": True,
        },
    },
    # =========================
    # laundry / utility
    # =========================
    "utility_sink": {
        "count": 4,
        "length_cm": (45, 65),
        "width_cm": (45, 70),
        "height_cm": (80, 95),
        "price_amount": (1200000, 8000000),
        "attributes": lambda rnd: {
            "freestanding": rnd.choice([True, False]),
        },
    },
    "ironing_board": {
        "count": 4,
        "length_cm": (5, 10),
        "width_cm": (35, 50),
        "height_cm": (110, 150),
        "price_amount": (250000, 2000000),
        "attributes": lambda rnd: {
            "foldable": True,
        },
    },
    "iron": {
        "count": 4,
        "length_cm": (12, 18),
        "width_cm": (10, 15),
        "height_cm": (20, 30),
        "price_amount": (300000, 2500000),
        "attributes": lambda rnd: {
            "steam": rnd.choice([True, False]),
        },
    },
    "drying_rack": {
        "count": 4,
        "length_cm": (40, 80),
        "width_cm": (90, 180),
        "height_cm": (90, 150),
        "price_amount": (300000, 2500000),
        "attributes": lambda rnd: {
            "foldable": rnd.choice([True, False]),
        },
    },
    # =========================
    # lighting / electrical
    # =========================
    "ceiling_light": {
        "count": 10,
        "length_cm": (8, 20),
        "width_cm": (25, 80),
        "height_cm": (25, 80),
        "price_amount": (400000, 6000000),
        "attributes": lambda rnd: {
            "mount": "ceiling",
            "type": rnd.choice(["flush", "semi_flush"]),
        },
    },
    "wall_sconce": {
        "count": 8,
        "length_cm": (10, 20),
        "width_cm": (12, 30),
        "height_cm": (20, 45),
        "price_amount": (300000, 3000000),
        "attributes": lambda rnd: {
            "mount": "wall",
        },
    },
    "track_light": {
        "count": 4,
        "length_cm": (8, 15),
        "width_cm": (60, 180),
        "height_cm": (8, 20),
        "price_amount": (1500000, 8000000),
        "attributes": lambda rnd: {
            "mount": "ceiling",
            "heads": rnd.choice([3, 4, 5, 6]),
        },
    },
    "table_lamp": {
        "count": 8,
        "length_cm": (18, 35),
        "width_cm": (18, 35),
        "height_cm": (35, 65),
        "price_amount": (300000, 3500000),
        "attributes": lambda rnd: {
            "mount": "table",
        },
    },
    "smart_speaker": {
        "count": 4,
        "length_cm": (8, 18),
        "width_cm": (8, 18),
        "height_cm": (10, 25),
        "price_amount": (800000, 5000000),
        "attributes": lambda rnd: {
            "voice_assistant": rnd.choice(["google", "alexa", "none"]),
        },
    },
    # =========================
    # decor / misc
    # =========================
    "clock": {
        "count": 6,
        "length_cm": (3, 8),
        "width_cm": (20, 60),
        "height_cm": (20, 60),
        "price_amount": (200000, 2500000),
        "attributes": lambda rnd: {
            "mount": rnd.choice(["wall", "table"]),
        },
    },
    "vase": {
        "count": 8,
        "length_cm": (10, 25),
        "width_cm": (10, 25),
        "height_cm": (20, 45),
        "price_amount": (150000, 2000000),
        "attributes": lambda rnd: {
            "material": rnd.choice(["ceramic", "glass", "metal"]),
        },
    },
    "cushion": {
        "count": 16,
        "length_cm": (8, 18),
        "width_cm": (35, 60),
        "height_cm": (35, 60),
        "price_amount": (100000, 800000),
        "attributes": lambda rnd: {
            "shape": rnd.choice(["square", "rect", "round"]),
        },
    },
    "throw_blanket": {
        "count": 10,
        "length_cm": (2, 6),
        "width_cm": (120, 180),
        "height_cm": (150, 220),
        "price_amount": (200000, 1800000),
        "attributes": lambda rnd: {
            "material": rnd.choice(["cotton", "wool", "fleece"]),
        },
    },
    "pet_bed": {
        "count": 4,
        "length_cm": (12, 30),
        "width_cm": (45, 100),
        "height_cm": (55, 120),
        "price_amount": (300000, 2500000),
        "attributes": lambda rnd: {
            "shape": rnd.choice(["oval", "rect", "round"]),
        },
    },
    # =========================
    # climate / utility / built-in
    # =========================
    "air_conditioner": {
        "count": 8,
        "length_cm": (18, 28),
        "width_cm": (70, 110),
        "height_cm": (25, 40),
        "price_amount": (6000000, 30000000),
        "attributes": lambda rnd: {
            "mount": "wall",
            "type": rnd.choice(["split", "cassette"]),
        },
    },
    "fan": {
        "count": 8,
        "length_cm": (25, 45),
        "width_cm": (25, 45),
        "height_cm": (90, 170),
        "price_amount": (500000, 5000000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["standing", "table", "tower"]),
        },
    },
    "heater": {
        "count": 4,
        "length_cm": (12, 30),
        "width_cm": (40, 80),
        "height_cm": (40, 70),
        "price_amount": (800000, 6000000),
        "attributes": lambda rnd: {
            "type": rnd.choice(["oil", "ceramic", "panel"]),
        },
    },
    "dehumidifier": {
        "count": 4,
        "length_cm": (20, 35),
        "width_cm": (25, 40),
        "height_cm": (40, 65),
        "price_amount": (2500000, 10000000),
        "attributes": lambda rnd: {
            "portable": True,
        },
    },
    "air_purifier": {
        "count": 4,
        "length_cm": (18, 35),
        "width_cm": (18, 35),
        "height_cm": (35, 70),
        "price_amount": (2500000, 12000000),
        "attributes": lambda rnd: {
            "portable": True,
        },
    },
}


SOFT_RULE_SNIPPETS = {
    "color_composition": (
        "Use a 60-30-10 balance with neutrals as base, controlled accents, and 2-5 main colors. "
        "Favor natural textures and reduce visual noise."
    ),
    "style_specific_aesthetics": (
        "Maintain material coherence and consistent shape language. Use layered textures instead of many colors."
    ),
    "room_specific_feng_shui": (
        "Keep the primary circulation path open, avoid blocking the entry line, and anchor key furniture to solid walls."
    ),
    "personal_feng_shui": (
        "Align accent colors with the user's element, avoid clashes with dominant materials, and keep balance between warm/cool."
    ),
    "lighting_color_interaction": (
        "Adjust palette based on daylight and CCT: warmer light needs warmer colors, bright daylight allows cooler neutrals."
    ),
}


def rand_int(rnd: random.Random, a: int, b: int) -> int:
    return int(rnd.randint(a, b))


def rand_price(rnd: random.Random, lo: int, hi: int) -> int:
    v = rnd.randint(lo, hi)
    return int(round(v / 10000) * 10000)


def _pick_color_for_type(
    rnd: random.Random, style_tags: list[str], type_name: str
) -> str:
    style = style_tags[0] if style_tags else "modern"
    system = STYLE_COLOR_SYSTEMS.get(style, STYLE_COLOR_SYSTEMS["modern"])
    role = _infer_color_role(type_name)
    palette = system.get(role) or system["neutral_misc"]
    return rnd.choice(palette)


def make_asset(rnd: random.Random, idx: int, type_name: str) -> Dict[str, Any]:
    spec = CATEGORY_SPECS[type_name]
    style_tags = rnd.sample(STYLES, k=rnd.randint(1, 2))
    material = rnd.choice(MATERIALS)
    brand = rnd.choice(BRANDS)

    length_cm = rand_int(rnd, *spec["length_cm"])
    width_cm = rand_int(rnd, *spec["width_cm"])
    height_cm = rand_int(rnd, *spec["height_cm"])
    length_mm = length_cm * 10
    width_mm = width_cm * 10
    height_mm = height_cm * 10
    price_amount = rand_price(rnd, *spec["price_amount"])

    attributes = spec["attributes"](rnd) if callable(spec.get("attributes")) else {}
    attributes.setdefault("color_hex", _pick_color_for_type(rnd, style_tags, type_name))
    attributes.setdefault("material_tags", [material])

    return {
        "id": f"{type_name}_{idx:03d}",
        "tenant_id": "demo_tenant",
        "type": type_name,
        "name": f"{type_name.replace('_', ' ').title()} {idx}",
        "style_tags": style_tags,
        "material": material,
        "brand": brand,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "price_amount": price_amount,
        "price_currency": CURRENCY,
        "attributes": attributes,
    }


def generate_inventory() -> List[Dict[str, Any]]:
    rnd = random.Random(SEED)
    inventory: List[Dict[str, Any]] = []

    for type_name, spec in CATEGORY_SPECS.items():
        for i in range(1, spec["count"] + 1):
            inventory.append(make_asset(rnd, i, type_name))

    return inventory


def generate_templates() -> List[Dict[str, Any]]:
    return [
        {
            "template_id": "tpl_bedroom_single_work",
            "name": "Bedroom: Single + Work Corner",
            "room_type": "bedroom",
            "style_tags": ["minimal", "japandi"],
            "zones": ["sleeping", "work", "storage"],
            "required_types": ["bed", "wardrobe"],
            "recommended_types": ["nightstand", "desk", "chair", "bookshelf"],
            "supported_shapes": ["rect", "L"],
            "layout_rules": [
                "Place bed against the longest wall, headboard to wall.",
                "Place wardrobe on a wall that does not block door swing and does not intersect window keep-out.",
                "Place desk near window wall but outside window keep-out zone; keep walkway >= 800mm.",
            ],
        },
        {
            "template_id": "tpl_bedroom_minimal",
            "name": "Bedroom: Minimal",
            "room_type": "bedroom",
            "style_tags": ["minimal", "scandinavian"],
            "zones": ["sleeping", "storage"],
            "required_types": ["bed", "wardrobe"],
            "recommended_types": ["nightstand", "bench"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Prioritize circulation: keep center area open.",
                "Keep at least one accessible side of bed >= 600mm clear.",
            ],
        },
        {
            "template_id": "tpl_bedroom_couple",
            "name": "Bedroom: Couple",
            "room_type": "bedroom",
            "style_tags": ["classic", "modern"],
            "zones": ["sleeping", "storage"],
            "required_types": ["bed", "wardrobe", "nightstand"],
            "recommended_types": ["dresser", "bench", "rug"],
            "supported_shapes": ["rect", "L"],
            "layout_rules": [
                "Place bed centered on main wall with two nightstands.",
                "Keep wardrobe clearance >= 900mm in front.",
            ],
        },
        {
            "template_id": "tpl_bedroom_kids",
            "name": "Bedroom: Kids Study",
            "room_type": "bedroom",
            "style_tags": ["modern", "scandinavian"],
            "zones": ["sleeping", "work", "storage"],
            "required_types": ["bed", "desk", "chair"],
            "recommended_types": ["bookshelf", "nightstand"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Desk near window for daylight; keep chair pull-out clearance.",
                "Keep play area open.",
            ],
        },
        {
            "template_id": "tpl_living_tv_sofa",
            "name": "Living Room: Sofa Facing TV",
            "room_type": "living_room",
            "style_tags": ["modern", "midcentury"],
            "zones": ["living", "storage"],
            "required_types": ["sofa", "tv_console"],
            "recommended_types": ["coffee_table", "bookshelf", "rug"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Place sofa to face TV console; avoid blocking windows and door swing.",
                "Keep main walkway >= 800mm from door to seating area.",
            ],
        },
        {
            "template_id": "tpl_living_conversation",
            "name": "Living Room: Conversation",
            "room_type": "living_room",
            "style_tags": ["boho", "coastal"],
            "zones": ["living"],
            "required_types": ["sofa", "armchair"],
            "recommended_types": ["coffee_table", "side_table", "rug", "floor_lamp"],
            "supported_shapes": ["rect", "L"],
            "layout_rules": [
                "Arrange sofa + armchairs around a coffee table.",
                "Keep seating area centered and avoid blocking windows.",
            ],
        },
        {
            "template_id": "tpl_living_open",
            "name": "Living Room: Open Plan",
            "room_type": "living_room",
            "style_tags": ["minimal", "japandi"],
            "zones": ["living", "storage"],
            "required_types": ["sofa"],
            "recommended_types": ["coffee_table", "side_table", "floor_lamp"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Use fewer large pieces, keep center open.",
                "Maintain clear path from entry to seating.",
            ],
        },
        {
            "template_id": "tpl_studio_compact",
            "name": "Studio: Compact Zoning",
            "room_type": "studio",
            "style_tags": ["modern", "minimal"],
            "zones": ["sleeping", "work", "living", "storage"],
            "required_types": ["bed", "wardrobe"],
            "recommended_types": [
                "desk",
                "chair",
                "sofa",
                "coffee_table",
                "tv_console",
            ],
            "supported_shapes": ["rect", "L"],
            "layout_rules": [
                "Split room into zones; avoid clutter.",
                "Keep window wall visually open; do not place tall items in window keep-out.",
            ],
        },
        {
            "template_id": "tpl_studio_zoned",
            "name": "Studio: Zoned Living",
            "room_type": "studio",
            "style_tags": ["japandi", "scandinavian"],
            "zones": ["sleeping", "living", "work"],
            "required_types": ["bed", "sofa"],
            "recommended_types": ["desk", "chair", "wardrobe", "rug"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Use rugs to separate zones.",
                "Keep circulation from door to window clear.",
            ],
        },
        {
            "template_id": "tpl_office_dual",
            "name": "Office: Dual Desks",
            "room_type": "office",
            "style_tags": ["modern", "industrial"],
            "zones": ["work", "storage"],
            "required_types": ["desk", "chair"],
            "recommended_types": ["bookshelf", "storage_cabinet"],
            "supported_shapes": ["rect", "L"],
            "layout_rules": [
                "Place desks to avoid glare and not block windows; keep chair pull-out clearance.",
                "Ensure circulation >= 800mm.",
            ],
        },
        {
            "template_id": "tpl_office_exec",
            "name": "Office: Executive",
            "room_type": "office",
            "style_tags": ["classic", "midcentury"],
            "zones": ["work", "meeting"],
            "required_types": ["desk", "chair"],
            "recommended_types": ["sofa", "coffee_table", "bookshelf"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Anchor desk against solid wall with view of entry.",
                "Keep meeting seating away from circulation path.",
            ],
        },
        {
            "template_id": "tpl_office_shared",
            "name": "Office: Shared Work",
            "room_type": "office",
            "style_tags": ["minimal", "scandinavian"],
            "zones": ["work"],
            "required_types": ["desk", "chair"],
            "recommended_types": ["storage_cabinet", "floor_lamp"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Keep desks parallel and maintain circulation.",
                "Ensure chair pull-out clearance.",
            ],
        },
        {
            "template_id": "tpl_kitchen_basic",
            "name": "Kitchen: Basic Line",
            "room_type": "kitchen",
            "style_tags": ["modern", "minimal"],
            "zones": ["cooking", "storage"],
            "required_types": ["kitchen_base_cabinet", "fridge"],
            "recommended_types": ["sink", "stove"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Place kitchen base cabinets along one wall; keep working aisle clearance.",
                "Do not block door swing; do not obstruct window keep-out.",
            ],
        },
        {
            "template_id": "tpl_kitchen_galley",
            "name": "Kitchen: Galley",
            "room_type": "kitchen",
            "style_tags": ["industrial", "modern"],
            "zones": ["cooking", "storage"],
            "required_types": ["kitchen_base_cabinet", "fridge"],
            "recommended_types": ["kitchen_wall_cabinet", "sink", "stove"],
            "supported_shapes": ["rect"],
            "layout_rules": [
                "Place base cabinets on two opposing walls.",
                "Maintain clear aisle between counters.",
            ],
        },
        {
            "template_id": "tpl_kitchen_island",
            "name": "Kitchen: Island",
            "room_type": "kitchen",
            "style_tags": ["modern", "coastal"],
            "zones": ["cooking", "storage"],
            "required_types": ["kitchen_base_cabinet", "kitchen_island", "fridge"],
            "recommended_types": ["sink", "stove"],
            "supported_shapes": ["rect", "L"],
            "layout_rules": [
                "Keep island clearance on all sides.",
                "Place sink or prep area in island if space allows.",
            ],
        },
    ]


def build_styling_docs() -> list[dict[str, Any]]:
    docs = [
        {
            "doc_id": "sty_global_color_balance",
            "title": "Global Color Balance",
            "tags": ["styling", "color", "global"],
            "text": (
                "Use a restrained whole-room palette. Let large furniture carry the base tones, "
                "medium pieces support them, and accessories provide small accents. "
                "A practical rule is 60-30-10: about 60 percent calm base colors, 30 percent secondary tones, "
                "and 10 percent accent color. Avoid giving every object a different strong color."
            ),
        },
        {
            "doc_id": "sty_global_material_consistency",
            "title": "Global Material Consistency",
            "tags": ["styling", "materials", "global"],
            "text": (
                "Keep 1 dominant material family and at most 1 or 2 supporting material families in one room. "
                "For example: wood + fabric as base, black metal as accent. "
                "Avoid mixing too many competing finishes unless the style explicitly allows it."
            ),
        },
        {
            "doc_id": "sty_global_visual_hierarchy",
            "title": "Visual Hierarchy in a Room",
            "tags": ["styling", "composition", "global"],
            "text": (
                "Choose one main visual anchor per zone. Large furniture such as bed, sofa, desk, or dining table "
                "should visually lead the room. Accessories should support the anchor instead of competing with it. "
                "Keep tall, dark, or high-contrast pieces limited."
            ),
        },
        {
            "doc_id": "sty_global_decor_density",
            "title": "Decor Density Control",
            "tags": ["styling", "decor", "global"],
            "text": (
                "Use decor sparingly. Empty space is part of the design. "
                "Prefer a few intentional accessories instead of many small scattered objects. "
                "If the room already has strong furniture shapes or strong colors, reduce extra decor."
            ),
        },
        {
            "doc_id": "sty_global_soft_vs_hard_balance",
            "title": "Soft and Hard Balance",
            "tags": ["styling", "soft_furnishing", "global"],
            "text": (
                "Balance hard surfaces such as wood, glass, and metal with soft elements such as rug, curtain, cushion, "
                "or fabric lampshades. Soft elements reduce visual harshness and help tie the palette together."
            ),
        },
        # ---------------- bedroom ----------------
        {
            "doc_id": "sty_bedroom_color_pairing",
            "title": "Bedroom Color Pairing",
            "tags": ["styling", "bedroom", "color", "bed", "nightstand", "wardrobe"],
            "text": (
                "In a bedroom, the bed should be the visual anchor. Keep bed, headboard, and rug in the same tonal family "
                "or closely related neutrals. Nightstands and lamps should support the bed, not outshine it. "
                "Wardrobes and dressers should usually stay quieter than the bed in both color and texture."
            ),
        },
        {
            "doc_id": "sty_bedroom_lamp_placement",
            "title": "Bedroom Lamp Placement",
            "tags": [
                "styling",
                "bedroom",
                "lamp",
                "nightstand",
                "bedside_lamp",
                "table_lamp",
            ],
            "text": (
                "Bedside lamps belong visually near the bed, usually on nightstands or visually paired beside it. "
                "Use symmetry for calm bedrooms and slight asymmetry for more casual rooms. "
                "Do not overload both sides with many decorative objects."
            ),
        },
        {
            "doc_id": "sty_bedroom_soft_layers",
            "title": "Bedroom Soft Layers",
            "tags": [
                "styling",
                "bedroom",
                "rug",
                "cushion",
                "throw_blanket",
                "mattress",
            ],
            "text": (
                "Use rug, cushions, and throw blanket as the main soft styling tools in a bedroom. "
                "Keep cushions and throws within the same palette as the bed, but allow one muted accent tone. "
                "Avoid more than one strong accent color on the bed."
            ),
        },
        # ---------------- living room ----------------
        {
            "doc_id": "sty_living_room_color_pairing",
            "title": "Living Room Color Pairing",
            "tags": [
                "styling",
                "living_room",
                "color",
                "sofa",
                "coffee_table",
                "tv_console",
            ],
            "text": (
                "In a living room, the sofa is usually the main color anchor. "
                "Coffee table, side table, and TV console should support the sofa with complementary neutrals or wood/metal contrast. "
                "Accent color should appear through cushions, art, or a small decor item rather than through many large objects."
            ),
        },
        {
            "doc_id": "sty_living_room_decor_placement",
            "title": "Living Room Decor Placement",
            "tags": [
                "styling",
                "living_room",
                "decor",
                "side_table",
                "console_table",
                "vase",
                "plant",
            ],
            "text": (
                "Place small decor where it supports existing furniture clusters: side table, console table, coffee table, or media shelf. "
                "Avoid spreading decor evenly everywhere. A single vase, one plant, or one lamp is often enough per support surface."
            ),
        },
        {
            "doc_id": "sty_living_room_lighting_layers",
            "title": "Living Room Lighting Layers",
            "tags": [
                "styling",
                "living_room",
                "lighting",
                "floor_lamp",
                "table_lamp",
                "wall_sconce",
            ],
            "text": (
                "Layer living room lighting. Ceiling light handles general light, while floor lamp or table lamp adds warmth near seating. "
                "Place lamps near sofa edges, armchairs, or side tables, not in the center of the room. "
                "Lighting should frame the seating group."
            ),
        },
        # ---------------- studio ----------------
        {
            "doc_id": "sty_studio_unified_palette",
            "title": "Studio Unified Palette",
            "tags": ["styling", "studio", "color", "zoning"],
            "text": (
                "A studio should use one unified palette across sleeping, working, and living zones. "
                "Different zones may vary in texture or accent strength, but not in completely different color families. "
                "Use rug, lamp, and decor to suggest zone identity without breaking visual unity."
            ),
        },
        {
            "doc_id": "sty_studio_zone_anchors",
            "title": "Studio Zone Anchors",
            "tags": ["styling", "studio", "bed", "desk", "sofa", "anchor"],
            "text": (
                "Let each studio zone have one primary anchor: bed for sleep, desk for work, sofa for lounge. "
                "Accessories should stay close to their anchor. Do not let decor from one zone visually spill too much into another."
            ),
        },
        # ---------------- office ----------------
        {
            "doc_id": "sty_office_color_pairing",
            "title": "Office Color Pairing",
            "tags": ["styling", "office", "color", "desk", "chair", "bookshelf"],
            "text": (
                "In an office, the desk is the main anchor. Keep desk, chair, and storage consistent in tone and material. "
                "Use a restrained palette to reduce distraction. Accent color should be small and intentional, such as art, lamp, or a single chair detail."
            ),
        },
        {
            "doc_id": "sty_office_accessory_control",
            "title": "Office Accessory Control",
            "tags": [
                "styling",
                "office",
                "desk",
                "laptop",
                "monitor",
                "desk_lamp",
                "decor",
            ],
            "text": (
                "Desk accessories should look functional first and decorative second. "
                "Prefer monitor, laptop, keyboard, mouse, and one desk lamp as the core set. "
                "Avoid adding many small decorative objects on the work surface."
            ),
        },
        # ---------------- kitchen ----------------
        {
            "doc_id": "sty_kitchen_color_pairing",
            "title": "Kitchen Color Pairing",
            "tags": [
                "styling",
                "kitchen",
                "color",
                "cabinet",
                "countertop",
                "appliance",
            ],
            "text": (
                "In a kitchen, cabinets define the main visual field. Keep large cabinet surfaces quiet and cohesive. "
                "Countertop appliances should be limited in visible color variety. "
                "If cabinets are strong in color, keep small appliances neutral. If cabinets are neutral, one accent appliance is acceptable."
            ),
        },
        {
            "doc_id": "sty_kitchen_countertop_styling",
            "title": "Kitchen Countertop Styling",
            "tags": [
                "styling",
                "kitchen",
                "countertop",
                "microwave",
                "coffee_machine",
                "kettle",
                "decor",
            ],
            "text": (
                "Countertop styling should remain sparse. Group functional appliances in one zone instead of scattering them. "
                "A small plant or one decorative object may soften the kitchen, but avoid crowding prep surfaces."
            ),
        },
        # ---------------- anchor-based decor ----------------
        {
            "doc_id": "sty_anchor_bed",
            "title": "Anchor Rule for Bed",
            "tags": [
                "styling",
                "anchor",
                "bed",
                "mattress",
                "headboard",
                "bedside_lamp",
                "rug",
            ],
            "text": (
                "The bed pairs best with mattress, headboard, rug, bedside lamp, cushions, and throw blanket. "
                "Keep bedside elements balanced. Rug should visually extend the bed zone rather than compete with it."
            ),
        },
        {
            "doc_id": "sty_anchor_desk",
            "title": "Anchor Rule for Desk",
            "tags": [
                "styling",
                "anchor",
                "desk",
                "laptop",
                "monitor",
                "keyboard",
                "mouse",
                "desk_lamp",
            ],
            "text": (
                "Desk accessories should look orderly and non-overlapping. A monitor or laptop is primary, a desk lamp is secondary. "
                "Keep left-right balance and reserve some empty desk surface."
            ),
        },
        {
            "doc_id": "sty_anchor_side_table",
            "title": "Anchor Rule for Side Table",
            "tags": [
                "styling",
                "anchor",
                "side_table",
                "table_lamp",
                "plant",
                "decor",
                "vase",
            ],
            "text": (
                "A side table usually needs only one main styled item, or two if both are small and visually different, "
                "for example lamp plus a small plant. Avoid stacking many decorative pieces on a small table."
            ),
        },
        {
            "doc_id": "sty_anchor_console_table",
            "title": "Anchor Rule for Console Table",
            "tags": [
                "styling",
                "anchor",
                "console_table",
                "mirror",
                "wall_art",
                "vase",
                "decor",
            ],
            "text": (
                "Console tables work best with a vertical visual partner such as mirror or wall art, plus one small tabletop object. "
                "Keep the tabletop composition centered or intentionally asymmetrical, but never crowded."
            ),
        },
        {
            "doc_id": "sty_anchor_sofa",
            "title": "Anchor Rule for Sofa",
            "tags": ["styling", "anchor", "sofa", "cushion", "throw_blanket", "rug"],
            "text": (
                "The sofa is styled mainly through cushions, throw blanket, nearby rug, and adjacent lighting. "
                "Use cushions to repeat the room palette, not introduce unrelated colors."
            ),
        },
        {
            "doc_id": "sty_anchor_tv_console",
            "title": "Anchor Rule for TV Console",
            "tags": ["styling", "anchor", "tv_console", "tv", "speaker", "decor"],
            "text": (
                "TV console styling should stay calm. TV is the main visual object. Speakers and small decor should stay secondary "
                "and should not create a busy media wall."
            ),
        },
        # ---------------- wall / ceiling / opening ----------------
        {
            "doc_id": "sty_wall_decor_composition",
            "title": "Wall Decor Composition",
            "tags": ["styling", "wall", "wall_art", "mirror", "clock", "wall_sconce"],
            "text": (
                "Wall decor should support furniture below it. Mirror, art, or clock should visually align with the anchor furniture group. "
                "Do not scatter many unrelated wall objects across every wall."
            ),
        },
        {
            "doc_id": "sty_window_treatment",
            "title": "Window Treatment Styling",
            "tags": ["styling", "window", "curtain", "blind", "color"],
            "text": (
                "Curtains and blinds should connect the room palette rather than stand out too much. "
                "Use window treatments to soften large walls and reinforce the main palette. "
                "They should usually be quieter than the main furniture anchor."
            ),
        },
        {
            "doc_id": "sty_ceiling_lighting_role",
            "title": "Ceiling Lighting Role",
            "tags": [
                "styling",
                "ceiling",
                "ceiling_light",
                "pendant_light",
                "track_light",
            ],
            "text": (
                "Ceiling lighting sets the room tone but should not dominate the color story. "
                "Pendant lights may act as a focal point over dining or island zones. "
                "Track lights and ceiling lights should remain visually coherent with hardware finishes in the room."
            ),
        },
    ]

    # ---- thêm toàn bộ docs màu theo style / role / item ----
    docs.extend(build_style_color_docs())

    return docs


def generate_design_kb_jsonl(path: str) -> None:
    docs = build_styling_docs()

    soft_templates = get_soft_rule_templates()
    for style in STYLES:
        palette = ", ".join(STYLE_PALETTES.get(style, STYLE_PALETTES["modern"]))
        materials = ", ".join(STYLE_MATERIAL_HINTS.get(style, []))
        lighting = STYLE_LIGHTING_HINTS.get(style, "balanced 3000K")
        color_system = STYLE_COLOR_SYSTEMS.get(style, STYLE_COLOR_SYSTEMS["modern"])

        docs.append(
            {
                "doc_id": f"style_core_{style}",
                "title": f"{style} core styling guidance",
                "tags": [
                    "style",
                    "styling",
                    style,
                    "palette",
                    "materials",
                    "lighting",
                    "color_system",
                ],
                "text": (
                    f"{style} style uses overall palette {palette}. "
                    f"Preferred materials: {materials}. "
                    f"Lighting character: {lighting}. "
                    f"Wall base colors: {', '.join(color_system['walls_main'])}. "
                    f"Wood floor tones: {', '.join(color_system['floor_wood'])}. "
                    f"Large furniture colors: {', '.join(color_system['large_casegoods'])}. "
                    f"Upholstery colors: {', '.join(color_system['upholstery'])}. "
                    f"Accent colors for small items only: {', '.join(color_system['accent_small'])}. "
                    f"Keep object colors cohesive across the whole room, "
                    f"use large furniture for base tones, and apply accent colors only through small accessories or soft furnishings."
                ),
            }
        )

        for room_type in ROOM_TYPES:
            docs.append(
                {
                    "doc_id": f"style_room_{style}_{room_type}",
                    "title": f"{style} styling for {room_type}",
                    "tags": [
                        "style",
                        "styling",
                        style,
                        room_type,
                        "room_specific",
                        "color",
                    ],
                    "text": (
                        f"For a {style} {room_type}, keep the main anchor furniture visually dominant and coordinate all secondary items around it. "
                        f"Wall colors should prefer {', '.join(color_system['walls_main'])}. "
                        f"Floor colors should prefer wood {', '.join(color_system['floor_wood'])} or stone {', '.join(color_system['floor_stone'])}. "
                        f"Large furniture should prefer {', '.join(color_system['large_casegoods'])}. "
                        f"Soft furnishings should prefer {', '.join(color_system['textiles'])}. "
                        f"Accent colors should stay small and come from {', '.join(color_system['accent_small'])}."
                    ),
                }
            )

        for template_id in soft_templates.keys():
            snippet = SOFT_RULE_SNIPPETS.get(
                template_id, "Keep solutions consistent with the style."
            )
            docs.append(
                {
                    "doc_id": f"soft_{template_id}_{style}",
                    "title": f"{style} {template_id} guidance",
                    "tags": ["soft_rule", "styling", template_id, style, "color"],
                    "text": (
                        f"{style} guidance: palette {palette}. Materials: {materials}. "
                        f"Lighting: {lighting}. "
                        f"Wall base: {', '.join(color_system['walls_main'])}. "
                        f"Floor tones: {', '.join(color_system['floor_wood'])}. "
                        f"Large furniture: {', '.join(color_system['large_casegoods'])}. "
                        f"Upholstery: {', '.join(color_system['upholstery'])}. "
                        f"Accent small items only: {', '.join(color_system['accent_small'])}. "
                        f"{snippet}"
                    ),
                }
            )

    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    inventory = generate_inventory()
    annotate_inventory_size_tiers(inventory)
    templates = generate_templates()

    with open(BASE_DIR / "inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    with open(BASE_DIR / "templates.json", "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

    generate_design_kb_jsonl(str(BASE_DIR / "design_kb.jsonl"))

    print("Wrote inventory.json, templates.json, design_kb.jsonl")


if __name__ == "__main__":
    main()
