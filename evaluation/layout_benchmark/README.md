# Layout Benchmark

This folder contains the evaluation process for comparing:

- `tknt_full`
- `idesign_openai`
- `flairgpt_openai`
- `tknt_a1_no_style_policy`
- `tknt_a2_no_capacity_control`
- `tknt_a3_single_concept`

The default benchmark set is `10` cases. With the 6 systems above, the full comparison has `60` system-case runs.
For TKNT-only component ablations, use `systems_ablation.json`.

## Fair Model Policy

Benchmark runs lock text generation to one model:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --model gpt-5.4-mini
```

The runner sets these env controls for TKNT subprocesses:

- `TKNT_BENCHMARK_TEXT_MODEL=<model>`: forces TKNT primary/helper/agent text models to the same model.
- `TKNT_DISABLE_MODEL_FALLBACK=1`: disables configured text-model fallback chains.
- TKNT now requires the style policy LLM output by default; no separate strict-mode flag is needed.
- Each TKNT/IDesign/FlairGPT subprocess starts from a fresh per-run `.cache`; IDesign also runs with `IDESIGN_DISABLE_CACHE=1`.

For IDesign and FlairGPT, run their prompts with the same locked text model and put outputs under `external_outputs`.
Their output wrapper includes `runtime_s`, `llm_usage`, `input_tokens`, and `output_tokens` when the OpenAI-compatible response includes usage data; the scorer preserves them in the summary.

Use separate keys for the external systems:

- IDesign: an OpenAI env file with `IDESIGN_USE_OPENAI=true` and either `IDESIGN_OPENAI_API_KEY` or `OPENAI_API_KEY`.
- FlairGPT: `other_systems/FlairGPT/Scene_Synthesis/hidden.env` with `FLAIRGPT_USE_OPENAI=true` and either `FLAIRGPT_OPENAI_API_KEY` or `OPENAI_API_KEY`.

## Commands

Create the run plan:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode plan
```

Run only TKNT systems:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --include-systems tknt_full,tknt_a1_no_style_policy,tknt_a2_no_capacity_control,tknt_a3_single_concept
```

Smoke the TKNT component ablation suite on two simple cases:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --systems evaluation/layout_benchmark/systems_ablation.json \
  --cases evaluation/layout_benchmark/simple_cases.json \
  --include-cases simple_living_02_compact_reading,simple_bedroom_01_rest_study \
  --model gpt-5.4-mini \
  --output-root evaluation/layout_benchmark/runs/ablation_smoke \
  --cases-root evaluation/layout_benchmark/tknt_case_artifacts/ablation_smoke
```

Run the full TKNT component ablation suite:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --systems evaluation/layout_benchmark/systems_ablation.json \
  --cases evaluation/layout_benchmark/cases.json \
  --model gpt-5.4-mini \
  --output-root evaluation/layout_benchmark/runs/ablation_full \
  --cases-root evaluation/layout_benchmark/tknt_case_artifacts/ablation_full
```

Run the TKNT stability holdout for full vs the currently competitive ablations:

```bash
./evaluation/layout_benchmark/run_stability_holdout.sh
```

This runs a 6-case subset from `stability_cases.json` once with `MODE=resume`
by default, so the first inspection pass only writes `r1`. To run both
non-deterministic repeats later, use:

```bash
REPEATS="r1 r2" ./evaluation/layout_benchmark/run_stability_holdout.sh
```

The default run writes:

- repeat outputs under `evaluation/layout_benchmark/runs/manual_stability/runs/r1`
- case artifacts under `evaluation/layout_benchmark/tknt_case_artifacts/manual_stability`
- stability analysis under `evaluation/layout_benchmark/runs/manual_stability`
- a `stability_evaluation` breakdown inside each per-case `score.json`

The holdout uses `systems_stability.json`: `tknt_full` plus
`no_style_policy`, `no_request_contract`, `no_tier_capacity`,
`no_material_macro_repair`, `no_stylist_llm`, and `no_gallery_selector`.
The default holdout narrows style coverage to three visually and spatially
distinct families: Japandi, Industrial, and Boho. The analyzer reports canonical
layout quality, style-layout behavior, palette fit, styling assignment
completeness, gallery diversity, reliability, a holistic stability score, and
paired full-vs-ablation non-inferiority statistics. Style scoring is
intentionally not based on style-specific asset tags such as "Japandi sofa" or
"industrial shelf"; it checks observable layout behavior such as solid-object
density, center openness, wall anchoring, decor density, daylight use, and
palette families.

Run the full automatic benchmark, including IDesign and FlairGPT:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --run-external \
  --model gpt-5.4-mini \
  --output-root evaluation/layout_benchmark/runs/main \
  --cases-root evaluation/layout_benchmark/tknt_case_artifacts/main
```

This executes all systems in `systems.json`, writes one `output.json` per
case/system under the selected `output-root`, and then scores the run.
Each system-case subprocess has a default 20 minute timeout. Timeout failures
are written as failed `output.json` files, then the runner continues with the
next system-case run. Override with `--timeout-s`; external systems inherit the
same limit unless `--external-timeout-s` is set.

Resume an interrupted run on the same `output-root`:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode resume \
  --run-external \
  --model gpt-5.4-mini \
  --output-root evaluation/layout_benchmark/runs/main \
  --cases-root evaluation/layout_benchmark/tknt_case_artifacts/main
```

With `run_one.sh`, set `MODE=resume` and keep the same `RUN_ROOT`, `CASES`,
`CASE`, and `SYSTEM` filters. Resume skips run directories whose `output.json`
can be read, then runs only missing or malformed outputs and writes
`resume_report.json`.

Score existing outputs:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode score
```

Quick smoke run:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run --max-runs 1
```

## External Outputs

Place each external result at one of these paths:

```text
evaluation/layout_benchmark/external_outputs/<system_id>/<case_id>/output.json
evaluation/layout_benchmark/external_outputs/<system_id>/<case_id>/output.txt
evaluation/layout_benchmark/external_outputs/<case_id>/<system_id>/output.json
evaluation/layout_benchmark/external_outputs/<case_id>/<system_id>/output.txt
```

`--mode plan` writes `pending_external_jobs.jsonl` with every expected path.

## Metrics

The scorer normalizes every system output into the same 2D schema, then reports:

- Runtime, input tokens, output tokens.
- Number of blocking objects out of bounds.
- Number and area of overlapping blocking object pairs.
- Opening clearance violations and door-to-center path blockers.
- Required object and relation following checks.
- Valid blocking object count, excluding out-of-bounds and overlapping blocking furniture.

The overall score is:

- 30% geometry validity.
- 25% request following.
- 25% arrangement reasonableness.
- 20% valid object count.

Decor/soft-overlay items such as rugs, curtains, cushions, wall art, and ceiling lights are not counted as blocking overlap violations.

For TKNT systems that generate multiple final variants, the benchmark records both the canonical variant selected by the pipeline and the best-scoring final-gallery variant. `scores.jsonl` uses the best final-gallery score for TKNT so generation quality can be compared independently from selector choice; `canonical_score` and `variant_evaluation.canonical_overall_score` preserve production behavior.

## Ablation Code Locations

The ablation switch lives in `pipeline/ablation_modes.py`. `no_capacity_control` remains as the legacy combined A2 mode, but clean component ablations should use `systems_ablation.json`.

| Mode | Component Tested | Behavior |
| --- | --- | --- |
| `full` | Baseline | Current TKNT pipeline. |
| `minimal_room_context` | Room interpreter context | Preserves geometry/openings/obstacles, strips derived affordance/guidance context before downstream planning. |
| `no_style_policy` | Style policy | Uses neutral/base style policy. |
| `no_semantic_layout_planner` | Seed relation planner semantic planner | Calls `SeedRelationPlanner.generate(..., use_semantic_planner=False)`. |
| `no_request_contract` | Request contract | Removes request-contract annotations before tier count, merge, solving, and refill. |
| `no_tier_capacity` | Tier/count capacity control | Forces zero/omitted tier decisions active before solving without skipping accessory refill. |
| `single_concept` | Concept count/gallery generation | Generates and solves one final concept. |
| `no_concept_repair` | Solver concept repair | Skips deterministic repair attempts after an UNSAT object-level solve. |
| `no_material_macro_repair` | Material macro diversity repair | Skips deterministic material-macro retargeting after gallery diversity collapse. |
| `no_accessory_refill` | Requested/decor refill | Skips requested non-functional refill and controlled accessory refill. |
| `no_stylist_llm` | Final stylist | Skips `generate_style_plan` and applies deterministic fallback styling. |
| `no_gallery_selector` | Final gallery selector | Keeps first candidates in solver order instead of distinct-gallery selection. |

When `tknt_full` is present, the runner also writes `ablation_summary.json` and `ablation_summary.csv` with deltas against `tknt_full`, canonical means, best-in-gallery means, runtime/tokens, failure count, variant count, and geometry/request metrics.

CLI usage:

```bash
python run_case_backend_cli.py --input path/to/input.json --ablation no_style_policy
```
