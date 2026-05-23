# Image Flow Eval

Standalone lab for testing TKNT image flows:

1. render an image from a 3D layout snapshot,
2. optionally edit that rendered image,
3. evaluate the output and ablation variants.

The lab intentionally excludes preview performance and WebGL reliability metrics.
It focuses on image validity, visual regression, reference similarity,
perceptual similarity, prompt alignment, batch distribution quality, and human
review summaries.

## Install

Use a local environment for this lab so the main project dependencies stay
stable:

```bash
python -m pip install -r /Users/Yuki/TKNT/experiments/image_flow_eval/requirements.txt
```

Optional metrics:

- LPIPS and FID require `torch` plus `torchmetrics`.
- CLIP similarity requires `open-clip-torch`.

If optional dependencies are missing, those metrics are reported as `skipped`
instead of crashing the run.

## Evaluate Existing Images

```bash
python -m experiments.image_flow_eval.image_flow_eval.cli evaluate \
  --manifest /Users/Yuki/TKNT/experiments/image_flow_eval/manifest.example.json \
  --output-dir /Users/Yuki/TKNT/experiments/image_flow_eval/runs/example_eval
```

Outputs:

- `summary.json`
- `summary.csv`
- per-case `metrics.json`
- visual diff images where visual metrics run

## Run Ablations Through TKNT Snapshot Renderer

Use this when a case has `snapshot_payload` and `snapshot_image`.

```bash
python -m experiments.image_flow_eval.image_flow_eval.cli run \
  --manifest /Users/Yuki/TKNT/experiments/image_flow_eval/manifest.example.json \
  --provider tknt_snapshot \
  --variants baseline no_layout_reference no_edit_mask no_preservation_prompt no_reference_images \
  --output-dir /Users/Yuki/TKNT/experiments/image_flow_eval/runs/tknt_snapshot_ablation
```

The `tknt_snapshot` provider uses the existing
`pipeline.snapshot_image_renderer.render_snapshot_image_from_path` function and
the repo's current image API environment.

## Command Provider

Use `provider: "command"` when you want to call an external renderer/editor.
The command can use these placeholders:

- `{case_id}`
- `{variant_id}`
- `{source_image}`
- `{reference_image}`
- `{mask_image}`
- `{output_image}`
- `{prompt}`

Example variant:

```json
{
  "id": "local_pix2pix",
  "provider": "command",
  "mode": "baseline",
  "command": [
    "python",
    "/Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_instruct_pix2pix.py",
    "--image",
    "{source_image}",
    "--prompt",
    "{prompt}",
    "--output",
    "{output_image}"
  ]
}
```

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

