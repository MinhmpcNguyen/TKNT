# Official Evaluation Results Package

Created: 2026-07-03T15:52:54

This folder collects the official/recognized evaluation artifacts from the requested run roots while excluding scratch outputs, caches, module internals, and unused loose runs.

## Selection Rules

### Layout Benchmark

Included top-level run groups whose names are `tknt_full` or start with `flairgpt_`, `idesign_`, or `tknt_new_`, as long as they contain `score.json` files. Loose one-case directories outside those official groups were not copied.

Copied artifacts include runner summaries, `score.json`, `input.json`, `output.json`, normalized layout files, variant score files, 2D layout images, status/token files, and lightweight diagnostics. Heavy `module_outputs/` and cache directories were intentionally excluded.

### Image Flow Create

Included only rows listed in `consolidated_core14_summary.csv/json` for `first_time` and `second_time`. The `curated_primary_create_batch1_current` folder was not copied because it is not part of the consolidated Core14 result set. Per-batch create `summary.*` files were not retained because the consolidated files are the official source of truth for create runs.

### Image Flow Edit

Included all official operation/batch `summary.csv/json` files under `edit_manual_by_operation_repeat1` and `edit_manual_by_operation_repeat2`, plus the variant directories and approved render sources referenced by those summaries.

## Quick Tables

- `tables/layout_scores_official.csv`
- `tables/layout_group_summary_official.csv`
- `tables/image_flow_create_core14_rows_official.csv`
- `tables/image_flow_edit_results_official.csv`
- `tables/image_flow_edit_metric_rows_official.csv`
- `tables/overview_counts.csv`

## Counts

- Layout groups: 31
- Layout score rows: 311
- Image-flow create rows: 168
- Image-flow edit result rows: 64
- Image-flow edit metric rows: 912
- Copied files: 4702
- Missing referenced files: 0

All copied source artifacts live under `files/`, preserving their original repository-relative paths.
