# Image Flow Eval

Standalone lab for testing TKNT image flows:

1. render an image from a 3D layout snapshot,
2. optionally edit that rendered image,
3. evaluate the output and ablation variants.

The lab intentionally excludes preview performance and WebGL reliability metrics.
It focuses on image validity, visual regression, pixel-level preservation
checks, ablation variants, and human review summaries.

## Install

Use a local environment for this lab so the main project dependencies stay
stable:

```bash
python -m pip install -r /Users/Yuki/TKNT/experiments/image_flow_eval/requirements.txt
```

Supported automated metrics:

- `blank`
- `visual_diff`
- `psnr`
- `ssim`
- `nrmse`
- `layout_fidelity`
- `layout_gate_score`
- `layout_edge_structure`
- `layout_depth_structure`
- `layout_dino_patch_similarity`
- `preset_fidelity`
- `layout_dino_object_similarity`
- `detector_layout`
- `detector_spatial_relations`

For create-layout preset evaluation, prefer metrics that are less sensitive to
finish changes caused by the selected style, lighting, and scenery presets:
`layout_gate_score`, gated `layout_fidelity`, `layout_edge_structure`,
`layout_depth_structure`, `layout_dino_patch_similarity`,
`layout_dino_object_similarity`, `detector_layout`,
`detector_spatial_relations`, and `preset_fidelity`. Avoid using `layout_lpips`
and global `layout_dino_similarity` as primary create metrics when presets are
enabled because lighting, scenery, and style can legitimately change global
appearance. `layout_depth_structure` uses the local
`TKNT_IMAGE_FLOW_DEPTH_MODEL` / Depth Anything cache and is skipped when the
model is not available locally.

## Evaluate Existing Images

```bash
python -m experiments.image_flow_eval.image_flow_eval.cli evaluate \
  --manifest /path/to/image-flow-manifest.json \
  --output-dir /Users/Yuki/TKNT/experiments/image_flow_eval/runs/example_eval
```

Outputs:

- `summary.json`
- `summary.csv`
- per-case `metrics.json`
- per-create `layout_fidelity_report.json` when `layout_fidelity` is enabled
- per-edit `render_layout_fidelity_report.json` and
  `edit_instruction_fidelity_report.json`
- visual diff images where visual metrics run

## Prepare Layout Snapshots

Use this to generate the 8 living-room/bedroom layout cases, open each layout in
the real 3D UI through a headless Chrome session, capture all camera presets,
choose the best-scored preset for the top-level snapshot, and save snapshot
inputs for later image generation/edit evaluation.

```bash
python -m experiments.image_flow_eval.prepare_layout_snapshots \
  --cases evaluation/layout_benchmark/cases.json \
  --include living_01_rect_sofa_tv,living_02_solid_wall_tv_window_clear,living_03_l_shape_circulation,living_04_large_sectional,bedroom_01_rect_door_right_window_left,bedroom_02_square_two_nightstands,bedroom_03_compact_study_window,bedroom_05_l_shape_storage \
  --output-root experiments/image_flow_eval/cases \
  --frontend-url http://127.0.0.1:5173 \
  --api-url http://127.0.0.1:8001
```

The command auto-starts the Vite frontend if `--frontend-url` is not already
reachable, with `VITE_ENABLE_EVAL_HOOKS=true`. If you start the frontend
yourself, use:

```bash
cd /Users/Yuki/TKNT/frontend
VITE_ENABLE_EVAL_HOOKS=true VITE_API_BASE=http://127.0.0.1:8001 npm run dev
```

Cases are processed end-to-end one at a time. After the log prints
`Snapshot ready: <case_id>`, that case directory already has the complete
layout/snapshot bundle even if a later case fails.

Per case, the command writes:

- `layout.json`
- `pipeline_output.json`
- `snapshot_payload.json`
- `snapshot.png`
- `snapshot_meta.json`
- `inventory_integrity_report.json` for imported benchmark runs
- `camera_snapshots/index.json`
- `camera_snapshots/*.png` and matching `*.snapshot_payload.json` for every
  captured camera preset

It also writes `manifest.generated.json` under the output root. This step does
not register layouts to an account and does not call the image generation/edit
API.

To snapshot existing layout benchmark runs without regenerating layouts, point
the command at one or more run-folder globs:

```bash
python -m experiments.image_flow_eval.prepare_layout_snapshots \
  --only-benchmark-runs \
  --benchmark-run-glob 'evaluation/layout_benchmark/runs/manual/runs/tknt_new_*' \
  --output-root experiments/image_flow_eval/cases \
  --frontend-url http://127.0.0.1:5173 \
  --api-url http://127.0.0.1:8001
```

Use `--max-camera-presets 8` or `--max-camera-presets 12` only when you want a
faster preview pass. The default captures all available camera presets so you can
review and discard bad angles manually.

After you have selected the best camera preset for each case, use
`--reuse-selected-camera-preset` to recapture only the preset already stored in
each case's `snapshot_meta.json`. Imported benchmark runs are rebuilt from their
own frozen `module_outputs` folder and fail before capture if the selected UI
layout inventory drifts from `output.json`.

## Run Ablations Through TKNT Snapshot Renderer

Use this when a case has `snapshot_payload` and `snapshot_image`.

```bash
python -m experiments.image_flow_eval.image_flow_eval.cli run \
  --manifest /path/to/image-flow-manifest.json \
  --provider tknt_snapshot \
  --variants simple_prompt contract_lite contract_objects operation_dsl no_preset_reference_images no_edit_mask \
  --output-dir /Users/Yuki/TKNT/experiments/image_flow_eval/runs/tknt_snapshot_ablation
```

The `tknt_snapshot` provider uses the existing
`pipeline.snapshot_image_renderer.render_snapshot_image_from_path` function and
the repo's current image API environment.

When `layout_fidelity` is enabled, the runner writes and evaluates the
aspect-aligned source artifacts saved beside each variant as `source_3d.png` and
`source_snapshot_payload.json`. The metric fails closed if the source/candidate
aspect differs, the vision evaluator cannot verify layout preservation, or any
required structured hard check is missing or failed. Edit variants separately
evaluate the approved render against the 3D source and the edited output against
that approved render. Raw vision judge scores are stored in
`details.raw_score`; the metric value is the effective score after hard-check
caps. `layout_gate_score` is the preferred ranking metric and keeps layout
failures capped even when preset, detector, or local structure metrics score
well.

Edit cases can reuse a create output that has already passed the layout gate:

```json
{
  "edit_source_variant": {
    "case_id": "living_room__create",
    "variant": "contract_objects_preflight"
  }
}
```

This avoids generating a new base render for every edit operation. Use the
stabilization manifest builder to generate the two lean benchmark manifests from
the current snapshot bbox values:

```bash
python -m experiments.image_flow_eval.build_stabilization_manifest \
  --profile compact_reading \
  --output /private/tmp/tknt_benchmark_1case_compact_reading_stabilization.json

python -m experiments.image_flow_eval.build_stabilization_manifest \
  --profile guest_vanity \
  --output /private/tmp/tknt_benchmark_1case_guest_vanity_lean.json
```

## Curated Create Suites

Generate the curated core-14 create suite, its 3-case batches, and the compact
preset-follow suite from `snapshot_audit/selected_snapshot_cases.json`:

```bash
python -m experiments.image_flow_eval.build_curated_create_manifests
```

Run one core batch:

```bash
BATCH=1
./experiments/image_flow_eval/run_image_flow_eval.sh run \
  --manifest experiments/image_flow_eval/snapshot_audit/manifest.curated_core14_create.batch${BATCH}.json \
  --provider tknt_snapshot
```

Recompute metrics without regenerating images:

```bash
BATCH=1
./experiments/image_flow_eval/run_image_flow_eval.sh evaluate \
  --manifest experiments/image_flow_eval/snapshot_audit/manifest.curated_core14_create.batch${BATCH}.json
```

The preset-follow suite crosses four representative core cases with three
shared style/lighting/scenery packs. Run the full compact matrix with the
leading variant after the core suite has identified it:

```bash
./experiments/image_flow_eval/run_image_flow_eval.sh run \
  --manifest experiments/image_flow_eval/snapshot_audit/manifest.curated_preset_follow_create.json \
  --provider tknt_snapshot \
  --variants simple_prompt
```

To run only one preset pack, use one of:
`japandi_day_garden`, `modern_night_city`, or
`industrial_overcast_mountain`.

## Human Ratings

Expected CSV columns:

- `case_id`
- `variant` or `variant_id`
- `rater`
- `score_instruction`
- `score_quality`
- `score_preservation`
- `artifact_severity`

```bash
python -m experiments.image_flow_eval.image_flow_eval.cli human-summary \
  --ratings /path/to/ratings.csv \
  --output /path/to/summary_human.json
```

## Ablation Modes

- `baseline`: normal flow.
- `simple_prompt`: forces the current simple prompt flow with text contract off
  and geometry/annotated/locked visual references off.
- `contract_lite`: keeps the same one-source-image flow, but adds a compact
  text-only scene contract without object rows.
- `contract_objects`: adds the text-only scene contract plus the top visible
  object map. It still does not send depth, edge, segmentation, structural guide,
  or geometry atlas images.
- `contract_objects_preflight`: combines `contract_objects` with a
  `gpt-5.4-mini` visual preflight grounded by snapshot metadata and room-shell
  locks.
- `contract_objects_preflight_visual_only`: keeps the object contract sent to the
  image model, but withholds metadata from `gpt-5.4-mini` to isolate preflight
  metadata grounding impact.
- `llm_visual_preflight`: classic visual preflight. It sends the clean source
  snapshot and compact metadata to `gpt-5.4-mini`, then appends object/opening
  preservation locks without deterministic room-shell/object-anchor overlays.
- `llm_visual_preflight_object_anchors`: extends `llm_visual_preflight` with
  room-shell locks, deterministic object bbox anchors, shadow-vs-geometry
  guidance, and a generated `room_shell_guide.png` annotated reference.
- `llm_visual_preflight_no_metadata`: same preflight controller, but withholds
  snapshot metadata from the LLM to isolate metadata grounding impact.
- `operation_dsl`: same as `contract_objects`, and edit prompts also include a
  structured operation contract for target locality.
- `operation_dsl_preflight`: combines `operation_dsl` with metadata-grounded
  visual preflight.
- `operation_dsl_preflight_no_dsl`: keeps visual preflight and the object
  contract but removes the structured edit operation DSL.
- `operation_dsl_preflight_no_edit_mask`: keeps visual preflight and the
  structured edit operation DSL but removes edit bboxes.
- `no_layout_reference`: removes layout and annotated reference images.
- `no_edit_mask`: removes edit bounding boxes from edit operations.
- `no_preservation_prompt`: weakens the user prompt to isolate preservation
  wording impact.
- `no_reference_images`: removes scene and replacement reference images.
- `no_preset_reference_images`: keeps text prompts and source images, but disables
  preset reference images to isolate their effect.

For visual preflight ablations, add the variants to a snapshot case:

```json
{
  "id": "case_create",
  "snapshot_payload": "case/snapshot_payload.json",
  "snapshot_image": "case/snapshot.png",
  "variants": [
    {"id": "llm_visual_preflight", "mode": "llm_visual_preflight"},
    {
      "id": "llm_visual_preflight_object_anchors",
      "mode": "llm_visual_preflight_object_anchors"
    },
    {
      "id": "llm_visual_preflight_no_metadata",
      "mode": "llm_visual_preflight_no_metadata"
    }
  ]
}
```

The preflight controller writes `visual_preflight_report.json` beside the
variant output and caches it by source-image, metadata, and model hash. It does
not retry image generation and does not change the default production flow.
