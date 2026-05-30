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

## Run Ablations Through TKNT Snapshot Renderer

Use this when a case has `snapshot_payload` and `snapshot_image`.

```bash
python -m experiments.image_flow_eval.image_flow_eval.cli run \
  --manifest /path/to/image-flow-manifest.json \
  --provider tknt_snapshot \
  --variants baseline no_layout_reference no_edit_mask no_preservation_prompt no_reference_images \
  --output-dir /Users/Yuki/TKNT/experiments/image_flow_eval/runs/tknt_snapshot_ablation
```

The `tknt_snapshot` provider uses the existing
`pipeline.snapshot_image_renderer.render_snapshot_image_from_path` function and
the repo's current image API environment.

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
- `no_layout_reference`: removes layout and annotated reference images.
- `no_edit_mask`: removes edit bounding boxes from edit operations.
- `no_preservation_prompt`: weakens the user prompt to isolate preservation
  wording impact.
- `no_reference_images`: removes scene and replacement reference images.
