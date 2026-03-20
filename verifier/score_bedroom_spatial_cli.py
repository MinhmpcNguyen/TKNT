from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verifier.bedroom_spatial_scorer import (  # noqa: E402
    load_stylist_output_json,
    score_bedroom_spatial,
)


def _flatten(obj: object, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, new_prefix))
        return out
    if isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
        return out
    out[prefix] = "" if obj is None else str(obj)
    return out


def _write_csv(path: Path, payload: dict[str, object]) -> None:
    flat = _flatten(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key in sorted(flat.keys()):
            writer.writerow([key, flat[key]])


def _find_latest_stylist(cases_root: Path) -> Path:
    if not cases_root.exists():
        raise SystemExit(f"Cases folder not found: {cases_root}")
    candidates = list(cases_root.glob("*/08_stylist.json"))
    if not candidates:
        raise SystemExit(f"No 08_stylist.json found under: {cases_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score bedroom spatial reasoning from a stylist output JSON")
    parser.add_argument(
        "--input",
        default="",
        help=(
            "Path to stylist output JSON (e.g. cases/<case_id>/08_stylist.json). "
            "If omitted, uses the newest cases/*/08_stylist.json."
        ),
    )
    parser.add_argument("--output", default="", help="Optional path to write the score JSON.")
    parser.add_argument(
        "--out-dir",
        default="bedroom",
        help="Output folder for default CSV/TXT (default: bedroom).",
    )
    args = parser.parse_args()

    input_path = (
        Path(args.input)
        if args.input
        else _find_latest_stylist(PROJECT_ROOT / "cases")
    )
    stylist_out = load_stylist_output_json(str(input_path))
    score = score_bedroom_spatial(stylist_out)

    score_payload = score.to_dict()
    rendered = json.dumps(score_payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        out_dir = Path(args.out_dir)
        _write_csv(out_dir / "score.csv", score_payload)
        (out_dir / "score.txt").write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
