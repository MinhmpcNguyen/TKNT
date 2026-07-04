from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.tier_count_director import TierCountDirector


def main() -> int:
    agent = TierCountDirector()

    description = ""
    special_notes = "Keep it uncluttered and airy."

    room_model_json = {
        "status": "OK",
        "room": {
            "unit": "mm",
            "room_id": "room_1",
            "name": "Main",
            "polygon_ccw": [
                {"x": 0, "y": 0},
                {"x": 6000, "y": 0},
                {"x": 6000, "y": 2000},
                {"x": 4000, "y": 2000},
                {"x": 4000, "y": 4000},
                {"x": 0, "y": 4000},
            ],
        },
        "openings": {
            "doors": [
                {
                    "id": "door_1",
                    "segment_mm": [{"x": 0, "y": 1600}, {"x": 0, "y": 2600}],
                    "swing_radius_mm": 900,
                    "hinge_hint": "LEFT",
                }
            ],
            "windows": [
                {
                    "id": "window_1",
                    "segment_mm": [{"x": 2000, "y": 0}, {"x": 4000, "y": 0}],
                    "clearance_mm": 300,
                }
            ],
        },
        "obstacles": [
            {
                "id": "door_1_swing",
                "type": "door_swing",
                "polygon_ccw": [
                    {"x": 0, "y": 1600},
                    {"x": 900, "y": 1600},
                    {"x": 900, "y": 2600},
                    {"x": 0, "y": 2600},
                ],
                "hard": True,
            },
            {
                "id": "window_1_clearance",
                "type": "window_clearance",
                "polygon_ccw": [
                    {"x": 2000, "y": 0},
                    {"x": 4000, "y": 0},
                    {"x": 4000, "y": 300},
                    {"x": 2000, "y": 300},
                ],
                "hard": True,
            },
        ],
        # "zones": [
        #     {
        #         "id": "zone_work",
        #         "tag": "work",
        #         "polygon_ccw": [
        #             {"x": 1000, "y": 1000},
        #             {"x": 3000, "y": 1000},
        #             {"x": 3000, "y": 3000},
        #             {"x": 1000, "y": 3000},
        #         ],
        #         "priority": 0.8,
        #     },
        #     {
        #         "id": "zone_circulation",
        #         "tag": "circulation",
        #         "polygon_ccw": [
        #             {"x": 0, "y": 0},
        #             {"x": 6000, "y": 0},
        #             {"x": 6000, "y": 4000},
        #             {"x": 0, "y": 4000},
        #         ],
        #         "priority": 0.5,
        #     },
        # ],
        "meta": {
            "room_type": "bedroom",
            "style": "japandi",
            "height_mm": 2800,
            "window_direction": "SE",
            "max_iterations": 12,
        },
        "notes": [],
        "missing": [],
        "conflicts": [],
    }

    user_intent_json = {
        "user_input": {
            "description": description,
            "special_description": special_notes,
        },
    }

    clusters_json = {
        "status": "OK",
        "clusters": [
            {
                "cluster_id": "sleep_area",
                "tag": "sleep",
                "members": ["bed", "nightstand"],
                "anchors": ["bed"],
                "hard_constraints": [
                    {"type": "no_overlap", "a": "bed", "b": "nightstand"},
                    {
                        "type": "anchor_side",
                        "a": "nightstand",
                        "b": "bed",
                        "side": "head_left",
                        "gap_min": 0,
                        "gap_max": 50,
                    },
                ],
                "soft_constraints": [],
                "cluster_rules": {
                    "grid_mm": 50,
                    "allowed_rotations": {
                        "bed": [0, 90],
                        "nightstand": [0, 90, 180, 270],
                    },
                    "facing": {
                        "bed": {
                            "front": "top",
                            "notes": "Headboard is at the bottom, foot of the bed faces top.",
                        },
                        "nightstand": {
                            "front": "top",
                            "notes": "Front of the nightstand faces the same direction as the bed.",
                        },
                    },
                    "access_requirements": [],
                },
            },
            {
                "cluster_id": "work_area",
                "tag": "work",
                "members": ["desk", "desk_chair"],
                "anchors": ["desk"],
                "hard_constraints": [
                    {"type": "no_overlap", "a": "desk", "b": "desk_chair"},
                    {
                        "type": "dock_to_edge",
                        "a": "desk_chair",
                        "b": "desk",
                        "b_edge": "front",
                        "span": "any",
                        "gap_min": 0,
                        "gap_max": 50,
                    },
                    {
                        "type": "requires_access",
                        "id": "desk",
                        "mode": "front_clearance",
                    },
                ],
                "soft_constraints": [],
                "cluster_rules": {
                    "grid_mm": 50,
                    "allowed_rotations": {
                        "desk": [0, 90],
                        "desk_chair": [0, 90, 180, 270],
                    },
                    "facing": {
                        "desk": {
                            "front": "top",
                            "notes": "Front of the desk is where the user sits.",
                        },
                        "desk_chair": {
                            "front": "top",
                            "notes": "Front of the chair faces the desk.",
                        },
                    },
                    "access_requirements": [
                        {
                            "id": "desk",
                            "type": "front_clearance",
                        }
                    ],
                },
            },
            {
                "cluster_id": "storage_area",
                "tag": "storage",
                "members": ["wardrobe_or_closet", "dresser", "book_shelf"],
                "anchors": ["wardrobe_or_closet"],
                "hard_constraints": [
                    {"type": "no_overlap", "a": "wardrobe_or_closet", "b": "dresser"},
                    {
                        "type": "no_overlap",
                        "a": "wardrobe_or_closet",
                        "b": "book_shelf",
                    },
                    {"type": "no_overlap", "a": "dresser", "b": "book_shelf"},
                    {
                        "type": "requires_access",
                        "id": "wardrobe_or_closet",
                        "mode": "front_clearance",
                    },
                    {
                        "type": "requires_access",
                        "id": "dresser",
                        "mode": "front_clearance",
                    },
                ],
                "soft_constraints": [],
                "cluster_rules": {
                    "grid_mm": 50,
                    "allowed_rotations": {
                        "wardrobe_or_closet": [0, 90],
                        "dresser": [0, 90],
                        "book_shelf": [0, 90],
                    },
                    "facing": {
                        "wardrobe_or_closet": {
                            "front": "top",
                            "notes": "Doors open towards the top.",
                        },
                        "dresser": {
                            "front": "top",
                            "notes": "Drawers open towards the top.",
                        },
                        "book_shelf": {
                            "front": "top",
                            "notes": "Books are accessed from the top.",
                        },
                    },
                    "access_requirements": [
                        {
                            "id": "wardrobe_or_closet",
                            "type": "front_clearance",
                        },
                        {
                            "id": "dresser",
                            "type": "front_clearance",
                        },
                    ],
                },
            },
        ],
        "notes": ["Clusters are organized based on typical bedroom functionality."],
        "missing": [],
    }

    try:
        result = agent.generate(
            description=description,
            special_notes=special_notes,
            room_model_json=room_model_json,
            user_intent_json=user_intent_json,
            clusters_json=clusters_json,
        )
    except Exception as exc:
        print("TEST_FAILED")
        print(str(exc))
        return 1

    print("RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "tier_count_director_output.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved result to: {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
