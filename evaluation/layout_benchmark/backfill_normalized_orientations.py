from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from layout.orientation_metadata import (
    enrich_normalized_layout_orientations,
    enrich_styled_output_orientations,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = BENCHMARK_DIR / "runs"

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillSummary:
    cases_seen: int
    cases_updated: int
    files_updated: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill TKNT orientation metadata from canonical absolute layouts into "
            "existing benchmark outputs and normalized JSON files."
        )
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Root containing benchmark run directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report updates without writing files.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    summary = backfill_runs(runs_root=Path(args.runs_root), dry_run=args.dry_run)
    _LOGGER.info(
        "Orientation backfill complete: %s cases scanned, %s cases updated, "
        "%s files updated.",
        summary.cases_seen,
        summary.cases_updated,
        summary.files_updated,
    )
    return 0


def backfill_runs(*, runs_root: Path, dry_run: bool) -> BackfillSummary:
    case_dirs = _tknt_case_dirs(runs_root)
    cases_updated = 0
    files_updated = 0
    for case_dir in case_dirs:
        updated = _backfill_case_dir(case_dir=case_dir, dry_run=dry_run)
        if updated:
            cases_updated += 1
            files_updated += updated
            _LOGGER.info("Updated %s file(s) under %s", updated, case_dir)
    return BackfillSummary(
        cases_seen=len(case_dirs),
        cases_updated=cases_updated,
        files_updated=files_updated,
    )


def _tknt_case_dirs(runs_root: Path) -> list[Path]:
    case_dirs = {
        path.parents[1]
        for path in runs_root.rglob("module_outputs/layout_variants.json")
    }
    case_dirs.update(path.parent for path in runs_root.rglob("09_layout_variants.json"))
    return sorted(case_dirs)


def _backfill_case_dir(*, case_dir: Path, dry_run: bool) -> int:
    variants_paths = tuple(
        path
        for path in (
            case_dir / "module_outputs" / "layout_variants.json",
            case_dir / "09_layout_variants.json",
        )
        if path.is_file()
    )
    if not variants_paths:
        return 0

    variants_payload = _load_json_mapping(variants_paths[0])
    absolute_layouts = _absolute_layouts_by_variant_id(variants_payload)
    canonical_layout = _canonical_absolute_layout(variants_payload)
    if not absolute_layouts and not canonical_layout:
        return 0

    files_updated = 0
    for path in variants_paths:
        files_updated += _update_json_file(
            path=path,
            dry_run=dry_run,
            transform=_enrich_layout_variants,
        )
    for path in (
        case_dir / "output.json",
        case_dir / "module_outputs" / "stylist.json",
        case_dir / "08_stylist.json",
    ):
        if path.is_file() and canonical_layout:
            files_updated += _update_json_file(
                path=path,
                dry_run=dry_run,
                transform=lambda payload: _enrich_canonical_output(
                    payload=payload,
                    canonical_layout=canonical_layout,
                ),
            )

    normalized_path = case_dir / "normalized.json"
    if normalized_path.is_file():
        files_updated += _update_json_file(
            path=normalized_path,
            dry_run=dry_run,
            transform=lambda payload: enrich_normalized_layout_orientations(
                normalized_payload=payload,
                absolute_layout_payload=_layout_for_normalized_payload(
                    payload=payload,
                    absolute_layouts=absolute_layouts,
                    canonical_layout=canonical_layout,
                ),
            ),
        )

    normalized_variants_path = case_dir / "normalized_variants.json"
    if normalized_variants_path.is_file():
        files_updated += _update_json_file(
            path=normalized_variants_path,
            dry_run=dry_run,
            transform=lambda payload: _enrich_normalized_variants(
                payload=payload,
                absolute_layouts=absolute_layouts,
                canonical_layout=canonical_layout,
            ),
        )
    return files_updated


def _enrich_layout_variants(payload: Mapping[str, object]) -> dict[str, object]:
    out = dict(payload)
    enriched_variants: list[object] = []
    for value in _sequence(payload.get("variants")):
        variant = _mapping(value)
        absolute_layout = _mapping(variant.get("absolute_layout"))
        styled_result = _mapping(variant.get("styled_result"))
        if not variant or not absolute_layout or not styled_result:
            enriched_variants.append(value)
            continue
        enriched_variant = dict(variant)
        enriched_variant["styled_result"] = enrich_styled_output_orientations(
            styled_payload=styled_result,
            absolute_layout_payload=absolute_layout,
        )
        enriched_variants.append(enriched_variant)
    out["variants"] = enriched_variants
    return out


def _enrich_canonical_output(
    *,
    payload: Mapping[str, object],
    canonical_layout: Mapping[str, object],
) -> dict[str, object]:
    final_output = _mapping(payload.get("final_output"))
    if final_output:
        return {
            **payload,
            "final_output": enrich_styled_output_orientations(
                styled_payload=final_output,
                absolute_layout_payload=canonical_layout,
            ),
        }
    return enrich_styled_output_orientations(
        styled_payload=payload,
        absolute_layout_payload=canonical_layout,
    )


def _enrich_normalized_variants(
    *,
    payload: Mapping[str, object],
    absolute_layouts: Mapping[str, Mapping[str, object]],
    canonical_layout: Mapping[str, object],
) -> dict[str, object]:
    out = dict(payload)
    enriched_variants: list[object] = []
    for value in _sequence(payload.get("variants")):
        variant = _mapping(value)
        if not variant:
            enriched_variants.append(value)
            continue
        enriched_variants.append(
            enrich_normalized_layout_orientations(
                normalized_payload=variant,
                absolute_layout_payload=_layout_for_normalized_payload(
                    payload=variant,
                    absolute_layouts=absolute_layouts,
                    canonical_layout=canonical_layout,
                ),
            )
        )
    out["variants"] = enriched_variants
    return out


def _layout_for_normalized_payload(
    *,
    payload: Mapping[str, object],
    absolute_layouts: Mapping[str, Mapping[str, object]],
    canonical_layout: Mapping[str, object],
) -> Mapping[str, object]:
    variant_id = _first_string(payload, "variant_id")
    return absolute_layouts.get(variant_id, canonical_layout)


def _absolute_layouts_by_variant_id(
    payload: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    layouts: dict[str, Mapping[str, object]] = {}
    for value in _sequence(payload.get("variants")):
        variant = _mapping(value)
        variant_id = _first_string(variant, "variant_id")
        absolute_layout = _mapping(variant.get("absolute_layout"))
        if variant_id and absolute_layout:
            layouts[variant_id] = absolute_layout
    return layouts


def _canonical_absolute_layout(payload: Mapping[str, object]) -> Mapping[str, object]:
    layouts = _absolute_layouts_by_variant_id(payload)
    selected_variant_id = _first_string(payload, "selected_variant_id")
    if selected_variant_id in layouts:
        return layouts[selected_variant_id]
    return next(iter(layouts.values()), {})


def _update_json_file(
    *,
    path: Path,
    dry_run: bool,
    transform: Callable[[Mapping[str, object]], dict[str, object]],
) -> int:
    payload = _load_json_mapping(path)
    updated = transform(payload)
    if updated == payload:
        return 0
    if not dry_run:
        path.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = _mapping(payload)
    if not item:
        raise ValueError(f"Expected a JSON object in {path}")
    return item


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _first_string(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
