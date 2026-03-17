from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster_composer.outline import compute_cluster_outline


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_for_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "cluster_id" in payload:
        return compute_cluster_outline(payload)

    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, val in payload.items():
            if isinstance(val, dict) and "cluster_id" in val:
                out[key] = compute_cluster_outline(val)
            else:
                out[key] = val
        return out

    raise ValueError("Unsupported JSON shape for cluster output")


def main() -> int:
    default_in = Path("/Users/Yuki/TKNT/test_agent/result/cluster_composer_all.json")
    default_out = Path("/Users/Yuki/TKNT/test_agent/result/cluster_outline_output.json")

    in_path = Path(input(f"Input JSON path (default: {default_in}): ") or default_in)
    out_path = Path(input(f"Output JSON path (default: {default_out}): ") or default_out)

    payload = _load_json(in_path)
    result = _run_for_payload(payload)
    _save_json(out_path, result)

    print(f"Saved outline result to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
