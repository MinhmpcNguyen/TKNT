# Layout Benchmark

This folder contains the evaluation process for comparing:

- `tknt_full`
- `idesign_openai`
- `flairgpt_openai`
- `tknt_a1_no_style_policy`
- `tknt_a2_no_capacity_control`
- `tknt_a3_single_concept`

The default benchmark set is `12` cases. With the 6 systems above, the full comparison has `72` system-case runs.

## Fair Model Policy

Benchmark runs lock text generation to one model:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --model gpt-5.4-nano
```

The runner sets these env controls for TKNT subprocesses:

- `TKNT_BENCHMARK_TEXT_MODEL=<model>`: forces TKNT primary/helper/agent text models to the same model.
- `TKNT_DISABLE_MODEL_FALLBACK=1`: disables configured text-model fallback chains.
- TKNT now requires the style policy LLM output by default; no separate strict-mode flag is needed.

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

Run the full automatic benchmark, including IDesign and FlairGPT:

```bash
python -m evaluation.layout_benchmark.run_benchmark --mode run \
  --run-external \
  --model gpt-5.4-nano \
  --output-root evaluation/layout_benchmark/runs/main \
  --cases-root evaluation/layout_benchmark/tknt_case_artifacts/main
```

This executes all systems in `systems.json`, writes one `output.json` per
case/system under the selected `output-root`, and then scores the run.

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

For systems that generate multiple candidates, the benchmark scores exactly one submitted final layout. TKNT full still creates its 5 internal concepts, but the scorer reads `final_output`, meaning the pipeline-selected canonical variant. There is no oracle best-of-5 scoring.

## Ablation Code Locations

The ablation switch lives in `pipeline/ablation_modes.py`.

- `full`: unchanged TKNT pipeline; style policy uses strict LLM during benchmark.
- `no_style_policy`: replaces compiled style policy with a neutral/base policy.
- `no_capacity_control`: disables tier-count capacity pruning/backoff effect by forcing omitted tier decisions active before solving and skipping accessory refill.
- `single_concept`: evaluates only the first final concept instead of the full 5-concept gallery.

CLI usage:

```bash
python run_case_backend_cli.py --input path/to/input.json --ablation no_style_policy
```
