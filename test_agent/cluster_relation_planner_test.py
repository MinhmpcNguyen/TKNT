from __future__ import annotations

import json
from pathlib import Path

from agent.cluster_relation_planner import ClusterRelationPlanner


def main() -> int:
    base = Path(__file__).resolve().parent / "result"
    room_path = base / "room_interpreter_output.json"
    outlines_path = base / "cluster_outline_output.json"

    if not room_path.exists():
        raise SystemExit(f"Missing {room_path}")
    if not outlines_path.exists():
        raise SystemExit(f"Missing {outlines_path}")

    room_model = json.loads(room_path.read_text(encoding="utf-8"))
    clusters = json.loads(outlines_path.read_text(encoding="utf-8"))

    planner = ClusterRelationPlanner()
    output = planner.generate(
        room_model_json=room_model,
        clusters_json=clusters,
        description="Auto-generated layout",
        special_notes="Keep circulation paths open near entry and windows.",
    )

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

