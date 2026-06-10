from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

DEFAULT_SYSTEMS = (
    "tknt_full",
    "tknt_ab_no_style_policy",
    "tknt_ab_no_request_contract",
    "tknt_ab_no_tier_capacity",
    "tknt_ab_no_material_macro_repair",
    "tknt_ab_no_stylist_llm",
    "tknt_ab_no_gallery_selector",
)
FULL_SYSTEM_ID = "tknt_full"
STYLE_FALLBACK_SCORE = 35.0
NON_BLOCKING_LAYOUT_TOKENS = (
    "rug",
    "carpet",
    "curtain",
    "blind",
    "lamp",
    "light",
    "cushion",
    "pillow",
    "throw",
    "art",
    "plant",
    "vase",
)
DECOR_LAYOUT_TOKENS = (
    "rug",
    "curtain",
    "blind",
    "lamp",
    "light",
    "cushion",
    "pillow",
    "throw",
    "art",
    "plant",
    "vase",
)
DAYLIGHT_SUPPORT_TOKENS = (
    "desk",
    "vanity",
    "armchair",
    "chair",
    "bookshelf",
    "bookcase",
    "floor_lamp",
    "lamp",
)
STYLE_LAYOUT_TARGETS: Mapping[str, Mapping[str, object]] = {
    "japandi": {
        "solid_count": (4.0, 8.0),
        "center_occupancy_max": 0.10,
        "wall_anchor_fraction": (0.45, 0.90),
        "decor_count": (1.0, 4.0),
        "daylight_support": True,
    },
    "industrial": {
        "solid_count": (5.0, 10.0),
        "center_occupancy_max": 0.18,
        "wall_anchor_fraction": (0.55, 1.0),
        "decor_count": (1.0, 5.0),
        "daylight_support": False,
    },
    "boho": {
        "solid_count": (5.0, 11.0),
        "center_occupancy_max": 0.22,
        "wall_anchor_fraction": (0.30, 0.90),
        "decor_count": (2.0, 7.0),
        "daylight_support": False,
    },
}
HOLISTIC_WEIGHTS: Mapping[str, float] = {
    "canonical_score": 0.50,
    "style_material_score": 0.25,
    "gallery_diversity_score": 0.15,
    "reliability_score": 0.10,
}
STYLE_MATERIAL_WEIGHTS: Mapping[str, float] = {
    "style_layout_score": 0.45,
    "palette_target_score": 0.35,
    "material_assignment_score": 0.20,
}
STYLE_LAYOUT_WEIGHTS: Mapping[str, float] = {
    "density_fit_score": 0.30,
    "center_openness_fit_score": 0.25,
    "wall_anchor_fit_score": 0.20,
    "decor_density_fit_score": 0.15,
    "daylight_usage_fit_score": 0.10,
}
GALLERY_DIVERSITY_WEIGHTS: Mapping[str, float] = {
    "distinct_variant_score": 0.75,
    "variant_count_score": 0.25,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze TKNT stability holdout runs with paired metrics."
    )
    parser.add_argument(
        "--run-roots",
        nargs="+",
        required=True,
        help="Benchmark output roots. Each root should contain runs/<case>__<system>.",
    )
    parser.add_argument(
        "--cases",
        default="evaluation/layout_benchmark/stability_cases.json",
        help="Stability cases JSON.",
    )
    parser.add_argument(
        "--systems",
        default="evaluation/layout_benchmark/systems_stability.json",
        help="Systems JSON used by the holdout.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/layout_benchmark/runs/manual_stability",
        help="Where stability summary artifacts are written.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20000,
        help="Bootstrap resamples for paired confidence intervals.",
    )
    parser.add_argument(
        "--permutation-reps",
        type=int,
        default=50000,
        help="Sign-flip permutation samples for paired p-values.",
    )
    args = parser.parse_args()

    cases = _case_by_id(Path(args.cases))
    systems = _system_ids(Path(args.systems))
    run_roots = [Path(row) for row in args.run_roots]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_rows = _collect_block_rows(
        run_roots=run_roots,
        cases=cases,
        systems=systems,
    )
    summary_rows = _summary_rows(block_rows=block_rows, systems=systems)
    pairwise_rows = _pairwise_rows(
        block_rows=block_rows,
        systems=systems,
        cases=cases,
        bootstrap_reps=max(1000, args.bootstrap_reps),
        permutation_reps=max(1000, args.permutation_reps),
    )

    _write_jsonl(output_dir / "stability_block_scores.jsonl", block_rows)
    _write_json(output_dir / "stability_summary.json", summary_rows)
    _write_csv(output_dir / "stability_summary.csv", summary_rows)
    _write_json(output_dir / "stability_pairwise.json", pairwise_rows)
    _write_csv(output_dir / "stability_pairwise.csv", pairwise_rows)

    print(
        f"Wrote stability block scores: {output_dir / 'stability_block_scores.jsonl'}"
    )
    print(f"Wrote stability summary: {output_dir / 'stability_summary.json'}")
    _print_console_summary(summary_rows=summary_rows, pairwise_rows=pairwise_rows)
    return 0


def _case_by_id(path: Path) -> dict[str, Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of cases in {path}")
    out: dict[str, Mapping[str, object]] = {}
    for row in payload:
        item = _mapping(row)
        case_id = _string(item.get("case_id") or item.get("id"))
        if case_id:
            out[case_id] = item
    if not out:
        raise ValueError(f"No cases found in {path}")
    return out


def _system_ids(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return DEFAULT_SYSTEMS
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return DEFAULT_SYSTEMS
    systems = [
        _string(_mapping(row).get("system_id"))
        for row in payload
        if _string(_mapping(row).get("system_id"))
    ]
    return tuple(systems or DEFAULT_SYSTEMS)


def _collect_block_rows(
    *,
    run_roots: Sequence[Path],
    cases: Mapping[str, Mapping[str, object]],
    systems: Sequence[str],
) -> list[dict[str, object]]:
    system_set = set(systems)
    rows: list[dict[str, object]] = []
    for index, run_root in enumerate(run_roots, start=1):
        repeat_id = run_root.name or f"r{index}"
        runs_dir = run_root / "runs"
        for score_path in sorted(runs_dir.glob("*__tknt*/score.json")):
            score = _read_mapping(score_path)
            system_id = _string(score.get("system_id"))
            case_id = _string(score.get("case_id"))
            if system_id not in system_set or case_id not in cases:
                continue
            run_dir = score_path.parent
            case = cases[case_id]
            stability_eval = _mapping(case.get("stability_eval"))
            canonical_score = _canonical_score(score)
            best_score = _best_score(score)
            variant_count = _variant_count(score)
            style_material = _style_material_score(
                run_dir=run_dir,
                score=score,
                stability_eval=stability_eval,
            )
            gallery = _gallery_score(
                run_dir=run_dir,
                score=score,
                stability_eval=stability_eval,
            )
            reliability = 0.0 if score.get("error") else 100.0
            holistic = _clamp_score(
                canonical_score * HOLISTIC_WEIGHTS["canonical_score"]
                + style_material["style_material_score"]
                * HOLISTIC_WEIGHTS["style_material_score"]
                + gallery["gallery_score"] * HOLISTIC_WEIGHTS["gallery_diversity_score"]
                + reliability * HOLISTIC_WEIGHTS["reliability_score"]
            )
            block_row = {
                "repeat_id": repeat_id,
                "case_id": case_id,
                "system_id": system_id,
                "run_dir": str(run_dir),
                "canonical_score": round(canonical_score, 3),
                "best_gallery_score": round(best_score, 3),
                "style_material_score": round(
                    style_material["style_material_score"], 3
                ),
                "style_family_score": round(style_material["style_family_score"], 3),
                "style_layout_score": round(style_material["style_layout_score"], 3),
                "density_fit_score": round(style_material["density_fit_score"], 3),
                "center_openness_fit_score": round(
                    style_material["center_openness_fit_score"], 3
                ),
                "wall_anchor_fit_score": round(
                    style_material["wall_anchor_fit_score"], 3
                ),
                "decor_density_fit_score": round(
                    style_material["decor_density_fit_score"], 3
                ),
                "daylight_usage_fit_score": round(
                    style_material["daylight_usage_fit_score"], 3
                ),
                "material_assignment_score": round(
                    style_material["material_assignment_score"], 3
                ),
                "material_target_score": round(
                    style_material["material_target_score"], 3
                ),
                "palette_target_score": round(
                    style_material["palette_target_score"], 3
                ),
                "material_coherence_score": round(
                    style_material["material_coherence_score"], 3
                ),
                "gallery_diversity_score": round(gallery["gallery_score"], 3),
                "gallery_distinct_count": gallery["distinct_count"],
                "gallery_variant_count": gallery["variant_count"],
                "gallery_target_count": gallery["target_count"],
                "variant_count": variant_count,
                "reliability_score": reliability,
                "holistic_stability_score": round(holistic, 3),
                "error": _string(score.get("error")),
                "target_style_family": _string(stability_eval.get("style_family")),
                "detected_style_family": style_material["detected_style_family"],
                "detected_material_families": sorted(
                    style_material["detected_material_families"]
                ),
                "detected_palette_families": sorted(
                    style_material["detected_palette_families"]
                ),
            }
            _write_score_stability_evaluation(
                score_path=score_path,
                score=score,
                block_row=block_row,
                stability_eval=stability_eval,
            )
            rows.append(block_row)
    return rows


def _write_score_stability_evaluation(
    *,
    score_path: Path,
    score: Mapping[str, object],
    block_row: Mapping[str, object],
    stability_eval: Mapping[str, object],
) -> None:
    stability_payload = _score_stability_evaluation_payload(
        block_row=block_row,
        stability_eval=stability_eval,
    )
    payload: dict[str, object] = {}
    inserted = False
    for key, value in score.items():
        if key == "stability_evaluation":
            continue
        payload[key] = value
        if key == "overall_score":
            payload["stability_evaluation"] = stability_payload
            inserted = True
    if not inserted:
        payload["stability_evaluation"] = stability_payload
    _write_json(score_path, payload)


def _score_stability_evaluation_payload(
    *,
    block_row: Mapping[str, object],
    stability_eval: Mapping[str, object],
) -> dict[str, object]:
    target_style = _normalize_token(stability_eval.get("style_family"))
    style_layout_target = dict(
        STYLE_LAYOUT_TARGETS.get(target_style) or STYLE_LAYOUT_TARGETS["japandi"]
    )
    return {
        "schema_version": 1,
        "source": "evaluation.layout_benchmark.analyze_stability_holdout",
        "metric_scope": (
            "Stability holdout metrics layered on top of the benchmark score. "
            "The benchmark component_scores remain unchanged."
        ),
        "repeat_id": _string(block_row.get("repeat_id")),
        "case_id": _string(block_row.get("case_id")),
        "system_id": _string(block_row.get("system_id")),
        "scores": {
            "holistic_stability_score": _float(
                block_row.get("holistic_stability_score")
            ),
            "canonical_score": _float(block_row.get("canonical_score")),
            "best_gallery_score": _float(block_row.get("best_gallery_score")),
            "style_material_score": _float(block_row.get("style_material_score")),
            "gallery_diversity_score": _float(block_row.get("gallery_diversity_score")),
            "reliability_score": _float(block_row.get("reliability_score")),
        },
        "weights": {
            "holistic_stability_score": dict(HOLISTIC_WEIGHTS),
            "style_material_score": dict(STYLE_MATERIAL_WEIGHTS),
            "style_layout_score": dict(STYLE_LAYOUT_WEIGHTS),
            "gallery_diversity_score": dict(GALLERY_DIVERSITY_WEIGHTS),
        },
        "formulas": {
            "holistic_stability_score": (
                "canonical_score * 0.50 + style_material_score * 0.25 + "
                "gallery_diversity_score * 0.15 + reliability_score * 0.10"
            ),
            "style_material_score": (
                "style_layout_score * 0.45 + palette_target_score * 0.35 + "
                "material_assignment_score * 0.20"
            ),
            "style_layout_score": (
                "density_fit_score * 0.30 + center_openness_fit_score * 0.25 + "
                "wall_anchor_fit_score * 0.20 + decor_density_fit_score * 0.15 + "
                "daylight_usage_fit_score * 0.10"
            ),
            "gallery_diversity_score": (
                "distinct_variant_score * 0.75 + variant_count_score * 0.25"
            ),
        },
        "style_material_breakdown": {
            "style_material_score": _float(block_row.get("style_material_score")),
            "style_layout_score": _float(block_row.get("style_layout_score")),
            "palette_target_score": _float(block_row.get("palette_target_score")),
            "material_assignment_score": _float(
                block_row.get("material_assignment_score")
            ),
        },
        "style_layout_breakdown": {
            "style_layout_score": _float(block_row.get("style_layout_score")),
            "density_fit_score": _float(block_row.get("density_fit_score")),
            "center_openness_fit_score": _float(
                block_row.get("center_openness_fit_score")
            ),
            "wall_anchor_fit_score": _float(block_row.get("wall_anchor_fit_score")),
            "decor_density_fit_score": _float(block_row.get("decor_density_fit_score")),
            "daylight_usage_fit_score": _float(
                block_row.get("daylight_usage_fit_score")
            ),
        },
        "gallery_breakdown": {
            "gallery_diversity_score": _float(block_row.get("gallery_diversity_score")),
            "distinct_count": int(_float(block_row.get("gallery_distinct_count"))),
            "variant_count": int(_float(block_row.get("gallery_variant_count"))),
            "target_count": int(_float(block_row.get("gallery_target_count"))),
        },
        "targets": {
            "style_family": target_style,
            "style_layout_target": style_layout_target,
            "target_palette_families": sorted(
                _string_sequence(stability_eval.get("target_palette_families"))
            ),
            "target_material_families": sorted(
                _string_sequence(stability_eval.get("target_material_families"))
            ),
            "gallery_min_distinct_variants": int(
                _float(stability_eval.get("gallery_min_distinct_variants"))
            ),
            "non_inferiority_margin": _float(
                stability_eval.get("non_inferiority_margin")
            ),
        },
        "detected": {
            "style_family": _string(block_row.get("detected_style_family")),
            "material_families": _string_sequence(
                block_row.get("detected_material_families")
            ),
            "palette_families": _string_sequence(
                block_row.get("detected_palette_families")
            ),
        },
        "diagnostic_scores_not_used_in_holistic": {
            "style_family_score": _float(block_row.get("style_family_score")),
            "material_target_score": _float(block_row.get("material_target_score")),
            "material_coherence_score": _float(
                block_row.get("material_coherence_score")
            ),
        },
        "notes": [
            "canonical_score is the production canonical variant score.",
            "best_gallery_score is reported for inspection but is not used in holistic_stability_score.",
            "style_family_score and material_target_score are diagnostic only in the current metric.",
            "Style scoring does not rely on style-specific asset tags.",
        ],
        "error": _string(block_row.get("error")),
    }


def _style_material_score(
    *,
    run_dir: Path,
    score: Mapping[str, object],
    stability_eval: Mapping[str, object],
) -> dict[str, object]:
    target_style = _normalize_token(stability_eval.get("style_family"))
    target_materials = set(
        _string_sequence(stability_eval.get("target_material_families"))
    )
    target_palette = set(
        _string_sequence(stability_eval.get("target_palette_families"))
    )
    variant = _canonical_layout_variant(run_dir=run_dir, score=score)
    final_style_plan = _mapping(
        _mapping(variant.get("styled_result")).get("final_style_plan")
    )
    detected_style = _detected_style_family(final_style_plan)
    objects = _canonical_objects(run_dir=run_dir, score=score, variant=variant)
    material_families = _material_families(objects)
    palette_families = _palette_families(objects, variant=variant)
    style_layout = _style_layout_score(
        run_dir=run_dir,
        score=score,
        target_style=target_style,
        objects=objects,
    )

    style_family_score = _style_family_score(
        target_style=target_style,
        detected_style=detected_style,
        final_style_plan=final_style_plan,
    )
    material_assignment_score = _material_assignment_score(
        objects=objects,
        material_families=material_families,
    )
    material_target_score = _coverage_score(material_families, target_materials)
    palette_target_score = _coverage_score(palette_families, target_palette)
    material_coherence_score = material_assignment_score
    style_material_score = _clamp_score(
        style_layout["style_layout_score"]
        * STYLE_MATERIAL_WEIGHTS["style_layout_score"]
        + palette_target_score * STYLE_MATERIAL_WEIGHTS["palette_target_score"]
        + material_assignment_score
        * STYLE_MATERIAL_WEIGHTS["material_assignment_score"]
    )
    return {
        "style_material_score": style_material_score,
        "style_family_score": style_family_score,
        "style_layout_score": style_layout["style_layout_score"],
        "density_fit_score": style_layout["density_fit_score"],
        "center_openness_fit_score": style_layout["center_openness_fit_score"],
        "wall_anchor_fit_score": style_layout["wall_anchor_fit_score"],
        "decor_density_fit_score": style_layout["decor_density_fit_score"],
        "daylight_usage_fit_score": style_layout["daylight_usage_fit_score"],
        "material_assignment_score": material_assignment_score,
        "material_target_score": material_target_score,
        "palette_target_score": palette_target_score,
        "material_coherence_score": material_coherence_score,
        "detected_style_family": detected_style,
        "detected_material_families": material_families,
        "detected_palette_families": palette_families,
    }


def _gallery_score(
    *,
    run_dir: Path,
    score: Mapping[str, object],
    stability_eval: Mapping[str, object],
) -> dict[str, object]:
    target = _int_or_default(stability_eval.get("gallery_min_distinct_variants"), 2)
    variants = _layout_variants(run_dir)
    if not variants:
        variants = _normalized_variants(run_dir)
    signatures = [_variant_signature(variant) for variant in variants]
    distinct_signatures = {signature for signature in signatures if signature}
    variant_count = len(variants)
    distinct_count = len(distinct_signatures)
    if variant_count <= 0:
        return {
            "gallery_score": 0.0,
            "variant_count": 0,
            "distinct_count": 0,
            "target_count": target,
        }
    count_score = min(100.0, 100.0 * variant_count / max(target, 1))
    distinct_score = min(100.0, 100.0 * distinct_count / max(target, 1))
    score_value = _clamp_score(
        distinct_score * GALLERY_DIVERSITY_WEIGHTS["distinct_variant_score"]
        + count_score * GALLERY_DIVERSITY_WEIGHTS["variant_count_score"]
    )
    return {
        "gallery_score": score_value,
        "variant_count": variant_count,
        "distinct_count": distinct_count,
        "target_count": target,
    }


def _summary_rows(
    *,
    block_rows: Sequence[Mapping[str, object]],
    systems: Sequence[str],
) -> list[dict[str, object]]:
    rows_by_system: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in block_rows:
        rows_by_system[_string(row.get("system_id"))].append(row)

    best_by_block: dict[tuple[str, str], float] = {}
    for row in block_rows:
        key = (_string(row.get("repeat_id")), _string(row.get("case_id")))
        best_by_block[key] = max(
            best_by_block.get(key, 0.0),
            _float(row.get("holistic_stability_score")),
        )

    out: list[dict[str, object]] = []
    for system_id in systems:
        rows = rows_by_system.get(system_id, [])
        if not rows:
            continue
        holistic = [_float(row.get("holistic_stability_score")) for row in rows]
        near_best_count = 0
        for row in rows:
            key = (_string(row.get("repeat_id")), _string(row.get("case_id")))
            if (
                best_by_block.get(key, 0.0)
                - _float(row.get("holistic_stability_score"))
                <= 2.0
            ):
                near_best_count += 1
        out.append(
            {
                "system_id": system_id,
                "block_count": len(rows),
                "mean_holistic_stability_score": round(_mean(holistic), 3),
                "median_holistic_stability_score": round(_median(holistic), 3),
                "p10_holistic_stability_score": round(_quantile(holistic, 0.10), 3),
                "worst_holistic_stability_score": round(min(holistic), 3),
                "std_holistic_stability_score": round(_stddev(holistic), 3),
                "mean_canonical_score": round(
                    _mean(_float(row.get("canonical_score")) for row in rows), 3
                ),
                "mean_best_gallery_score": round(
                    _mean(_float(row.get("best_gallery_score")) for row in rows), 3
                ),
                "mean_style_material_score": round(
                    _mean(_float(row.get("style_material_score")) for row in rows), 3
                ),
                "mean_gallery_diversity_score": round(
                    _mean(_float(row.get("gallery_diversity_score")) for row in rows),
                    3,
                ),
                "failure_count": sum(1 for row in rows if _string(row.get("error"))),
                "near_best_within_2pt_count": near_best_count,
                "near_best_within_2pt_rate": round(near_best_count / len(rows), 3),
            }
        )
    return sorted(
        out,
        key=lambda row: _float(row.get("mean_holistic_stability_score")),
        reverse=True,
    )


def _pairwise_rows(
    *,
    block_rows: Sequence[Mapping[str, object]],
    systems: Sequence[str],
    cases: Mapping[str, Mapping[str, object]],
    bootstrap_reps: int,
    permutation_reps: int,
) -> list[dict[str, object]]:
    rows_by_block: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(
        dict
    )
    for row in block_rows:
        key = (_string(row.get("repeat_id")), _string(row.get("case_id")))
        rows_by_block[key][_string(row.get("system_id"))] = row

    metrics = {
        "holistic_stability_score": "holistic_stability_score",
        "canonical_score": "canonical_score",
        "style_material_score": "style_material_score",
        "gallery_diversity_score": "gallery_diversity_score",
    }
    raw_rows: list[dict[str, object]] = []
    p_full_values: list[tuple[int, float]] = []
    p_ablation_values: list[tuple[int, float]] = []

    for system_id in systems:
        if system_id == FULL_SYSTEM_ID:
            continue
        common_blocks = [
            (key, rows)
            for key, rows in rows_by_block.items()
            if FULL_SYSTEM_ID in rows and system_id in rows
        ]
        if not common_blocks:
            continue
        margin = _non_inferiority_margin(
            cases,
            [case_id for (_repeat_id, case_id), _rows in common_blocks],
        )
        for metric_name, row_key in metrics.items():
            diffs = [
                _float(rows[FULL_SYSTEM_ID].get(row_key))
                - _float(rows[system_id].get(row_key))
                for _key, rows in common_blocks
            ]
            mean_delta = _mean(diffs)
            ci_low, ci_high = _bootstrap_ci(
                diffs,
                reps=bootstrap_reps,
                seed=17,
            )
            p_full_greater, p_ablation_greater = _paired_permutation_one_sided(
                diffs,
                observed=mean_delta,
                reps=permutation_reps,
                seed=29,
            )
            row = {
                "system_id": system_id,
                "metric": metric_name,
                "block_count": len(diffs),
                "mean_delta_full_minus_ablation": round(mean_delta, 3),
                "ci95_low": round(ci_low, 3),
                "ci95_high": round(ci_high, 3),
                "full_win_count": sum(1 for value in diffs if value > 0),
                "ablation_win_count": sum(1 for value in diffs if value < 0),
                "tie_count": sum(1 for value in diffs if value == 0),
                "non_inferiority_margin": margin,
                "full_non_inferior": ci_low >= -margin,
                "full_better": ci_low > 0,
                "ablation_better": ci_high < 0,
                "p_full_greater": p_full_greater,
                "p_ablation_greater": p_ablation_greater,
                "holm_p_full_greater": None,
                "holm_p_ablation_greater": None,
            }
            row_index = len(raw_rows)
            raw_rows.append(row)
            if metric_name == "holistic_stability_score":
                p_full_values.append((row_index, p_full_greater))
                p_ablation_values.append((row_index, p_ablation_greater))

    adjusted_full = _holm_adjust(p_full_values)
    for row_index, p_value in adjusted_full.items():
        raw_rows[row_index]["holm_p_full_greater"] = p_value
    adjusted_ablation = _holm_adjust(p_ablation_values)
    for row_index, p_value in adjusted_ablation.items():
        raw_rows[row_index]["holm_p_ablation_greater"] = p_value
    return raw_rows


def _canonical_score(score: Mapping[str, object]) -> float:
    variant_evaluation = _mapping(score.get("variant_evaluation"))
    value = variant_evaluation.get("canonical_overall_score")
    return _float(value if value is not None else score.get("overall_score"))


def _best_score(score: Mapping[str, object]) -> float:
    variant_evaluation = _mapping(score.get("variant_evaluation"))
    value = variant_evaluation.get("best_overall_score")
    return _float(value if value is not None else score.get("overall_score"))


def _variant_count(score: Mapping[str, object]) -> int:
    variant_evaluation = _mapping(score.get("variant_evaluation"))
    return _int_or_default(variant_evaluation.get("variant_count"), 1)


def _canonical_layout_variant(
    *,
    run_dir: Path,
    score: Mapping[str, object],
) -> Mapping[str, object]:
    variants = _layout_variants(run_dir)
    if not variants:
        return {}
    canonical_id = _canonical_variant_id(score)
    if canonical_id:
        for variant in variants:
            if _string(variant.get("variant_id")) == canonical_id:
                return variant
    return variants[0]


def _canonical_objects(
    *,
    run_dir: Path,
    score: Mapping[str, object],
    variant: Mapping[str, object],
) -> list[Mapping[str, object]]:
    styled_result = _mapping(variant.get("styled_result"))
    objects = _sequence(styled_result.get("objects"))
    if objects:
        return [_mapping(row) for row in objects if _mapping(row)]

    normalized_variants = _normalized_variants(run_dir)
    canonical_id = _canonical_variant_id(score)
    for normalized in normalized_variants:
        if canonical_id and _string(normalized.get("variant_id")) != canonical_id:
            continue
        rows = [
            _object_payload(row)
            for row in _sequence(normalized.get("objects"))
            if _object_payload(row)
        ]
        if rows:
            return rows
    normalized = _read_mapping(run_dir / "normalized.json")
    return [
        _object_payload(row)
        for row in _sequence(normalized.get("objects"))
        if _object_payload(row)
    ]


def _canonical_variant_id(score: Mapping[str, object]) -> str:
    variant_evaluation = _mapping(score.get("variant_evaluation"))
    return _string(variant_evaluation.get("canonical_variant_id"))


def _layout_variants(run_dir: Path) -> list[Mapping[str, object]]:
    payload = _read_json(run_dir / "module_outputs" / "layout_variants.json")
    root = _mapping(payload)
    rows = _sequence(root.get("variants")) if root else _sequence(payload)
    return [_mapping(row) for row in rows if _mapping(row)]


def _normalized_variants(run_dir: Path) -> list[Mapping[str, object]]:
    payload = _read_mapping(run_dir / "normalized_variants.json")
    return [
        _mapping(row) for row in _sequence(payload.get("variants")) if _mapping(row)
    ]


def _variant_signature(variant: Mapping[str, object]) -> str:
    parts = [
        _string(variant.get("state_signature")),
        _string(variant.get("macro_layout_signature")),
        _string(variant.get("actual_macro_layout_signature")),
        _string(variant.get("planned_material_macro_signature")),
        _geometry_signature(variant),
    ]
    return "|".join(part for part in parts if part)


def _geometry_signature(variant: Mapping[str, object]) -> str:
    styled_result = _mapping(variant.get("styled_result"))
    objects = _sequence(styled_result.get("objects")) or _sequence(
        variant.get("objects")
    )
    rows: list[str] = []
    for row in objects:
        item = _object_payload(row)
        if not item:
            continue
        object_type = _normalize_token(item.get("object_type") or item.get("type"))
        bbox = _mapping(item.get("bbox"))
        if not bbox:
            polygon = _sequence(item.get("polygon_ccw") or item.get("polygon"))
            bbox = _bbox_from_polygon(polygon)
        if not object_type or not bbox:
            continue
        rows.append(
            ":".join(
                [
                    object_type,
                    str(round(_float(bbox.get("min_x")) / 500)),
                    str(round(_float(bbox.get("min_y")) / 500)),
                    str(round(_float(bbox.get("max_x")) / 500)),
                    str(round(_float(bbox.get("max_y")) / 500)),
                ]
            )
        )
    return ",".join(sorted(rows))


def _object_payload(row: object) -> Mapping[str, object]:
    item = _mapping(row)
    raw = _mapping(item.get("raw"))
    return raw or item


def _detected_style_family(final_style_plan: Mapping[str, object]) -> str:
    style_name = _normalize_token(final_style_plan.get("style_name"))
    if style_name:
        return style_name
    trace = _mapping(final_style_plan.get("layout_policy_trace"))
    return _normalize_token(trace.get("style_name"))


def _style_family_score(
    *,
    target_style: str,
    detected_style: str,
    final_style_plan: Mapping[str, object],
) -> float:
    if not target_style:
        return 100.0
    if detected_style == target_style:
        return 100.0
    tags = {
        _normalize_token(tag)
        for tag in _sequence(final_style_plan.get("style_tags"))
        if _normalize_token(tag)
    }
    if target_style in tags:
        return 80.0
    if detected_style in {"balanced", "minimal"} and target_style in {
        "japandi",
        "scandinavian",
        "minimal",
    }:
        return 55.0
    if not detected_style:
        return STYLE_FALLBACK_SCORE
    return 20.0


def _style_layout_score(
    *,
    run_dir: Path,
    score: Mapping[str, object],
    target_style: str,
    objects: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    target = STYLE_LAYOUT_TARGETS.get(target_style) or STYLE_LAYOUT_TARGETS["japandi"]
    normalized_layout = _canonical_normalized_layout(run_dir=run_dir, score=score)
    room_bbox = _bbox_from_polygon(_sequence(normalized_layout.get("room_polygon")))
    object_rows = list(objects)
    if not object_rows:
        object_rows = [
            _object_payload(row)
            for row in _sequence(normalized_layout.get("objects"))
            if _object_payload(row)
        ]

    solid_rects: list[Mapping[str, object]] = []
    decor_count = 0
    wall_anchor_count = 0
    daylight_candidates: list[Mapping[str, object]] = []
    for row in object_rows:
        object_type = _normalize_token(
            row.get("object_type") or row.get("type") or row.get("id")
        )
        bbox = _object_bbox(row)
        if not bbox:
            continue
        if _is_decor_layout_object(object_type):
            decor_count += 1
        if not _is_solid_layout_object(object_type):
            continue
        solid_rects.append(bbox)
        if room_bbox and _is_wall_anchored(bbox=bbox, room_bbox=room_bbox):
            wall_anchor_count += 1
        if any(token in object_type for token in DAYLIGHT_SUPPORT_TOKENS):
            daylight_candidates.append(bbox)

    solid_count = float(len(solid_rects))
    wall_anchor_fraction = wall_anchor_count / max(len(solid_rects), 1)
    density_fit = _range_fit_score(
        solid_count,
        _float_pair(target.get("solid_count"), default=(4.0, 9.0)),
    )
    center_openness = _center_openness_fit_score(
        room_bbox=room_bbox,
        solid_rects=solid_rects,
        max_occupancy=_float(target.get("center_occupancy_max") or 0.14),
    )
    wall_anchor_fit = _range_fit_score(
        wall_anchor_fraction,
        _float_pair(target.get("wall_anchor_fraction"), default=(0.40, 0.90)),
    )
    decor_density_fit = _range_fit_score(
        float(decor_count),
        _float_pair(target.get("decor_count"), default=(1.0, 5.0)),
    )
    daylight_usage = _daylight_usage_fit_score(
        normalized_layout=normalized_layout,
        daylight_candidates=daylight_candidates,
        required=bool(target.get("daylight_support")),
    )
    style_layout = _clamp_score(
        density_fit * STYLE_LAYOUT_WEIGHTS["density_fit_score"]
        + center_openness * STYLE_LAYOUT_WEIGHTS["center_openness_fit_score"]
        + wall_anchor_fit * STYLE_LAYOUT_WEIGHTS["wall_anchor_fit_score"]
        + decor_density_fit * STYLE_LAYOUT_WEIGHTS["decor_density_fit_score"]
        + daylight_usage * STYLE_LAYOUT_WEIGHTS["daylight_usage_fit_score"]
    )
    return {
        "style_layout_score": style_layout,
        "density_fit_score": density_fit,
        "center_openness_fit_score": center_openness,
        "wall_anchor_fit_score": wall_anchor_fit,
        "decor_density_fit_score": decor_density_fit,
        "daylight_usage_fit_score": daylight_usage,
    }


def _canonical_normalized_layout(
    *,
    run_dir: Path,
    score: Mapping[str, object],
) -> Mapping[str, object]:
    canonical_id = _canonical_variant_id(score)
    for row in _normalized_variants(run_dir):
        if canonical_id and _string(row.get("variant_id")) != canonical_id:
            continue
        return row
    return _read_mapping(run_dir / "normalized.json")


def _object_bbox(row: Mapping[str, object]) -> Mapping[str, object]:
    item = _object_payload(row)
    bbox = _mapping(item.get("bbox"))
    if bbox:
        return bbox
    return _bbox_from_polygon(_sequence(item.get("polygon_ccw") or item.get("polygon")))


def _is_solid_layout_object(object_type: str) -> bool:
    return bool(object_type) and not any(
        token in object_type for token in NON_BLOCKING_LAYOUT_TOKENS
    )


def _is_decor_layout_object(object_type: str) -> bool:
    return any(token in object_type for token in DECOR_LAYOUT_TOKENS)


def _is_wall_anchored(
    *,
    bbox: Mapping[str, object],
    room_bbox: Mapping[str, object],
) -> bool:
    threshold = 450.0
    return (
        abs(_float(bbox.get("min_x")) - _float(room_bbox.get("min_x"))) <= threshold
        or abs(_float(bbox.get("max_x")) - _float(room_bbox.get("max_x"))) <= threshold
        or abs(_float(bbox.get("min_y")) - _float(room_bbox.get("min_y"))) <= threshold
        or abs(_float(bbox.get("max_y")) - _float(room_bbox.get("max_y"))) <= threshold
    )


def _center_openness_fit_score(
    *,
    room_bbox: Mapping[str, object],
    solid_rects: Sequence[Mapping[str, object]],
    max_occupancy: float,
) -> float:
    if not room_bbox:
        return 70.0
    min_x = _float(room_bbox.get("min_x"))
    min_y = _float(room_bbox.get("min_y"))
    max_x = _float(room_bbox.get("max_x"))
    max_y = _float(room_bbox.get("max_y"))
    width = max_x - min_x
    depth = max_y - min_y
    if width <= 0 or depth <= 0:
        return 70.0
    center_rect = {
        "min_x": min_x + width * 0.30,
        "max_x": min_x + width * 0.70,
        "min_y": min_y + depth * 0.30,
        "max_y": min_y + depth * 0.70,
    }
    center_area = _rect_area(center_rect)
    occupied = sum(_rect_intersection_area(rect, center_rect) for rect in solid_rects)
    occupancy = occupied / max(center_area, 1.0)
    if occupancy <= max_occupancy:
        return 100.0
    return _clamp_score(100.0 - (occupancy - max_occupancy) * 350.0)


def _daylight_usage_fit_score(
    *,
    normalized_layout: Mapping[str, object],
    daylight_candidates: Sequence[Mapping[str, object]],
    required: bool,
) -> float:
    if not required:
        return 100.0
    windows = [
        opening
        for opening in _sequence(normalized_layout.get("openings"))
        if _string(_mapping(opening).get("kind")) == "window"
    ]
    if not windows:
        return 75.0
    if not daylight_candidates:
        return 50.0
    window_points = [_opening_midpoint(_mapping(opening)) for opening in windows]
    best_distance = min(
        _distance(_rect_center(candidate), window_point)
        for candidate in daylight_candidates
        for window_point in window_points
    )
    if best_distance <= 1800.0:
        return 100.0
    if best_distance <= 2800.0:
        return 70.0
    return 35.0


def _range_fit_score(value: float, target_range: tuple[float, float]) -> float:
    lower, upper = target_range
    if lower <= value <= upper:
        return 100.0
    distance = lower - value if value < lower else value - upper
    span = max(upper - lower, 1.0)
    return _clamp_score(100.0 - 55.0 * distance / span)


def _float_pair(value: object, *, default: tuple[float, float]) -> tuple[float, float]:
    rows = list(_sequence(value))
    if len(rows) != 2:
        return default
    return _float(rows[0]), _float(rows[1])


def _rect_area(rect: Mapping[str, object]) -> float:
    return max(0.0, _float(rect.get("max_x")) - _float(rect.get("min_x"))) * max(
        0.0,
        _float(rect.get("max_y")) - _float(rect.get("min_y")),
    )


def _rect_intersection_area(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> float:
    width = max(
        0.0,
        min(_float(first.get("max_x")), _float(second.get("max_x")))
        - max(_float(first.get("min_x")), _float(second.get("min_x"))),
    )
    depth = max(
        0.0,
        min(_float(first.get("max_y")), _float(second.get("max_y")))
        - max(_float(first.get("min_y")), _float(second.get("min_y"))),
    )
    return width * depth


def _rect_center(rect: Mapping[str, object]) -> tuple[float, float]:
    return (
        (_float(rect.get("min_x")) + _float(rect.get("max_x"))) / 2.0,
        (_float(rect.get("min_y")) + _float(rect.get("max_y"))) / 2.0,
    )


def _opening_midpoint(opening: Mapping[str, object]) -> tuple[float, float]:
    start = _mapping(opening.get("start"))
    end = _mapping(opening.get("end"))
    return (
        (_float(start.get("x")) + _float(end.get("x"))) / 2.0,
        (_float(start.get("y")) + _float(end.get("y"))) / 2.0,
    )


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _material_families(objects: Sequence[Mapping[str, object]]) -> set[str]:
    families: set[str] = set()
    for row in objects:
        row_families: set[str] = set()
        text = _string(row.get("material")).lower()
        color_family = _color_family(_string(row.get("color_hex")))
        if any(
            token in text
            for token in ("wood", "oak", "ash", "walnut", "plywood", "reclaimed")
        ):
            row_families.add("wood")
        if any(
            token in text
            for token in (
                "fabric",
                "linen",
                "cotton",
                "wool",
                "textile",
                "upholstery",
                "velvet",
                "canvas",
            )
        ):
            row_families.add("fabric")
        if any(
            token in text for token in ("woven", "rattan", "wicker", "cane", "jute")
        ):
            row_families.add("woven")
        if any(
            token in text
            for token in ("metal", "steel", "iron", "aluminum", "brushed", "blackened")
        ):
            row_families.add("metal")
        if any(token in text for token in ("brass", "bronze", "gold")):
            row_families.add("brass")
        if "leather" in text:
            row_families.add("leather")
        if any(token in text for token in ("stone", "marble", "concrete", "plaster")):
            row_families.add("stone")
        if "metal" in row_families and color_family == "black":
            row_families.add("black_metal")
        if color_family == "wood":
            row_families.add("wood")
        families.update(row_families)
    return families


def _palette_families(
    objects: Sequence[Mapping[str, object]],
    *,
    variant: Mapping[str, object],
) -> set[str]:
    families: set[str] = set()
    for row in objects:
        color = _string(row.get("color_hex"))
        family = _color_family(color)
        if family:
            families.add(family)
    styled_result = _mapping(variant.get("styled_result"))
    room = _mapping(styled_result.get("room"))
    surfaces = _mapping(room.get("surfaces"))
    for value in surfaces.values():
        family = _color_family(_string(value))
        if family:
            families.add(family)
    if "wood" in _material_families(objects):
        families.add("wood")
    return families


def _color_family(color_hex: str) -> str:
    text = color_hex.strip().lstrip("#")
    if len(text) != 6:
        return ""
    try:
        red = int(text[0:2], 16) / 255.0
        green = int(text[2:4], 16) / 255.0
        blue = int(text[4:6], 16) / 255.0
    except ValueError:
        return ""
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    hue_degrees = hue * 360.0
    if value < 0.22:
        return "black"
    if saturation < 0.12 and value > 0.82:
        return "white"
    if saturation < 0.16:
        return "grey"
    if 18.0 <= hue_degrees <= 48.0 and saturation >= 0.22:
        if saturation >= 0.45 and value >= 0.42:
            return "terracotta"
        return "wood"
    if 48.0 < hue_degrees <= 72.0 and saturation >= 0.25:
        return "brass"
    if 75.0 <= hue_degrees <= 165.0:
        return "green"
    if saturation < 0.32 and value > 0.58:
        return "warm_neutral"
    return "warm_neutral"


def _coverage_score(actual: set[str], target: set[str]) -> float:
    if not target:
        return 100.0
    return _clamp_score(100.0 * len(actual & target) / len(target))


def _material_assignment_score(
    *,
    objects: Sequence[Mapping[str, object]],
    material_families: set[str],
) -> float:
    if not objects:
        return 0.0
    material_rows = [
        row
        for row in objects
        if _string(row.get("material")) or _string(row.get("color_hex"))
    ]
    coverage = len(material_rows) / max(len(objects), 1)
    variety = min(len(material_families), 3) / 3.0
    return _clamp_score(coverage * 80.0 + variety * 20.0)


def _non_inferiority_margin(
    cases: Mapping[str, Mapping[str, object]],
    case_ids: Sequence[str],
) -> float:
    margins = []
    for case_id in case_ids:
        stability_eval = _mapping(cases.get(case_id, {}).get("stability_eval"))
        margins.append(_float(stability_eval.get("non_inferiority_margin") or 2.0))
    return round(_mean(margins), 3)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    count = len(values)
    samples = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(reps)
    ]
    samples.sort()
    return samples[int(0.025 * reps)], samples[min(reps - 1, int(0.975 * reps))]


def _paired_permutation_one_sided(
    values: Sequence[float],
    *,
    observed: float,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 1.0, 1.0
    rng = random.Random(seed)
    count = len(values)
    full_greater = 0
    ablation_greater = 0
    for _ in range(reps):
        sampled = (
            sum(value if rng.random() < 0.5 else -value for value in values) / count
        )
        if sampled >= observed - 1e-12:
            full_greater += 1
        if sampled <= observed + 1e-12:
            ablation_greater += 1
    denom = reps + 1
    return (full_greater + 1) / denom, (ablation_greater + 1) / denom


def _holm_adjust(indexed_p_values: Sequence[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(indexed_p_values, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (row_index, p_value) in enumerate(ordered):
        value = min(1.0, (total - index) * p_value)
        running = max(running, value)
        adjusted[row_index] = round(running, 6)
    return adjusted


def _bbox_from_polygon(points: Sequence[object]) -> Mapping[str, object]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        row = _mapping(point)
        if not row:
            continue
        xs.append(_float(row.get("x")))
        ys.append(_float(row.get("y")))
    if not xs or not ys:
        return {}
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _print_console_summary(
    *,
    summary_rows: Sequence[Mapping[str, object]],
    pairwise_rows: Sequence[Mapping[str, object]],
) -> None:
    print("\nTop systems by holistic stability:")
    for row in summary_rows[:8]:
        print(
            f"{_string(row.get('system_id'))}: "
            f"mean={_float(row.get('mean_holistic_stability_score')):.3f}, "
            f"p10={_float(row.get('p10_holistic_stability_score')):.3f}, "
            f"worst={_float(row.get('worst_holistic_stability_score')):.3f}, "
            f"fail={int(_float(row.get('failure_count')))}"
        )

    print("\nFull vs ablation on holistic stability:")
    for row in pairwise_rows:
        if _string(row.get("metric")) != "holistic_stability_score":
            continue
        print(
            f"{_string(row.get('system_id'))}: "
            f"delta={_float(row.get('mean_delta_full_minus_ablation')):.3f}, "
            f"ci=[{_float(row.get('ci95_low')):.3f},{_float(row.get('ci95_high')):.3f}], "
            f"noninferior={bool(row.get('full_non_inferior'))}, "
            f"ablation_better_holm_p={row.get('holm_p_ablation_greater')}"
        )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_mapping(path: Path) -> Mapping[str, object]:
    return _mapping(_read_json(path))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["system_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list | tuple) else ()


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _string_sequence(value: object) -> tuple[str, ...]:
    return tuple(
        token for item in _sequence(value) if (token := _normalize_token(item))
    )


def _normalize_token(value: object) -> str:
    return _string(value).lower().replace("-", "_").replace(" ", "_")


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int_or_default(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / max(len(rows), 1)


def _median(values: Sequence[float]) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    mid = len(rows) // 2
    if len(rows) % 2:
        return rows[mid]
    return (rows[mid - 1] + rows[mid]) / 2.0


def _quantile(values: Sequence[float], quantile: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    index = max(0, min(len(rows) - 1, math.ceil(len(rows) * quantile) - 1))
    return rows[index]


def _stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


if __name__ == "__main__":
    raise SystemExit(main())
