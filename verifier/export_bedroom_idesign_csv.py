from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _find_score_file(folder: Path) -> Path | None:
    for name in ("report_score.txt", "score.txt"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _oob_passed(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    status = str(value.get("status") or "").lower()
    if not status:
        return None
    return status == "passed"


def _bbl_passed(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    mean = value.get("mean_intersection_area_mm2")
    if mean is None:
        return None
    try:
        return float(mean) == 0.0
    except (TypeError, ValueError):
        return None


def _numeric_recall(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    recall = value.get("recall")
    if recall is None:
        return None
    try:
        return float(recall)
    except (TypeError, ValueError):
        return None


def _collect_columns(root: Path, prefix: str) -> tuple[list[str], list[Any]]:
    columns: list[str] = []
    values: list[Any] = []

    for folder in sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    ):
        score_file = _find_score_file(folder)
        if score_file is None:
            continue
        data = _read_json(score_file)
        idx = folder.name

        columns.extend(
            [
                f"{prefix}_{idx}_spatial_score",
                f"{prefix}_{idx}_OOB",
                f"{prefix}_{idx}_BBL",
                f"{prefix}_{idx}_numeric_score_recall",
            ]
        )
        values.extend(
            [
                data.get("overall_score"),
                _oob_passed(data.get("oob")),
                _bbl_passed(data.get("bbl")),
                _numeric_recall(data.get("numerical_reasoning")),
            ]
        )

    return columns, values


def main() -> int:
    root = Path(__file__).resolve().parent / "bedroom"
    columns, values = _collect_columns(root, "bedroom")

    output_path = root / "TKNT.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", *columns])
        writer.writerow(["TKNT", *values])

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
