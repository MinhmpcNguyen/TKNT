from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cluster_placer import MacroClusterPlacer


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Test MacroClusterPlacer agent")
    parser.add_argument(
        "--room",
        default=str(REPO_ROOT / "test_agent" / "result" / "room_interpreter_output.json"),
        help="Path to room_model_json",
    )
    parser.add_argument(
        "--clusters",
        default=str(REPO_ROOT / "test_agent" / "result" / "cluster_outline_output.json"),
        help="Path to clusters_outlines_json",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "test_agent" / "result" / "cluster_placer_output.json"),
        help="Output path",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=50,
        help="grid_mm for placement",
    )
    args = parser.parse_args()

    room_path = Path(args.room)
    clusters_path = Path(args.clusters)
    out_path = Path(args.out)

    if not room_path.exists():
        print(f"Missing room model file: {room_path}")
        return 1
    if not clusters_path.exists():
        print(f"Missing clusters outline file: {clusters_path}")
        return 1

    room_model_json = _load_json(room_path)
    clusters_outlines_json = _load_json(clusters_path)

    placer = MacroClusterPlacer()
    result = placer.generate(
        room_model_json=room_model_json,
        clusters_outlines_json=clusters_outlines_json,
        grid_mm=args.grid,
    )

    _save_json(out_path, result)
    print("RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved result to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
