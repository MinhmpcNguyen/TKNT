from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

build_solver_object_program = importlib.import_module(
    "object_program.builder"
).build_solver_object_program


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    result_dir = Path(__file__).resolve().parent / "result"
    cluster_path = result_dir / "seed_relation_planner_output.json"
    if not cluster_path.exists():
        cluster_path = result_dir / "cluster_forge_output.json"
    tier_path = result_dir / "tier_count_director_output.json"

    if not cluster_path.exists():
        print(f"Missing: {cluster_path}")
        return 1
    if not tier_path.exists():
        print(f"Missing: {tier_path}")
        return 1

    seed_relation_plan = _load_json(cluster_path)
    tier_count = _load_json(tier_path)

    merged = build_solver_object_program(seed_relation_plan, tier_count)

    output_path = result_dir / "cluster_merged_output.json"
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved merged output to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
