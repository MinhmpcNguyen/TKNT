from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.cluster_composer import ClusterComposer


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_one(agent: ClusterComposer, merged: dict, cluster_id: str) -> tuple[str, dict]:
    result = agent.generate(
        merged_clusters=merged,
        cluster_id=cluster_id,
        description="",
        special_notes="",
    )
    return cluster_id, result


def main() -> int:
    result_dir = Path(__file__).resolve().parent / "result"
    merged_path = result_dir / "cluster_merged_output.json"
    if not merged_path.exists():
        print(f"Missing: {merged_path}")
        return 1

    merged = _load_json(merged_path)
    clusters = merged.get("clusters", [])
    if not isinstance(clusters, list) or not clusters:
        print("No clusters found in merged output.")
        return 1

    cluster_ids = [c.get("cluster_id") for c in clusters if isinstance(c, dict)]
    cluster_ids = [cid for cid in cluster_ids if isinstance(cid, str) and cid]
    if not cluster_ids:
        print("No valid cluster_id values.")
        return 1

    agent = ClusterComposer()
    outputs: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=min(3, len(cluster_ids))) as executor:
        futures = [executor.submit(_run_one, agent, merged, cid) for cid in cluster_ids]
        for future in as_completed(futures):
            try:
                cid, payload = future.result()
            except Exception as exc:
                print(f"FAILED cluster_id: {exc}")
                continue
            outputs[cid] = payload
            out_path = result_dir / f"cluster_composer_{cid}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved {cid} -> {out_path}")

    merged_out_path = result_dir / "cluster_composer_all.json"
    merged_out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved all outputs -> {merged_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
