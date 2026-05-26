# Hugging Face Image Edit Lab for macOS

This folder provides two standalone Diffusers runners for local experiments on
macOS:

- `black-forest-labs/FLUX.2-klein-base-4B`
- `black-forest-labs/FLUX.1-schnell`
- `timbrooks/instruct-pix2pix`

The scripts are written for Apple Silicon Macs first, but they also support
CUDA and CPU fallback.

## Why a separate lab

- Previous local image-edit experiments showed that local behavior was not
  reliable enough for exact-preservation testing.
- These scripts let you test the Hugging Face Diffusers pipelines directly.
- The code adapts device and dtype automatically for macOS.

## Important notes

- `FLUX.2-klein-base-4B` is still a heavy model. Black Forest Labs says it fits
  in about 13 GB VRAM on NVIDIA GPUs. On macOS, it may still be slow or run out
  of memory depending on your machine.
- `InstructPix2Pix` is the more realistic first thing to try on macOS.
- Your current `python3` environment does not have `torch`, `diffusers`,
  `transformers`, or `Pillow` installed yet.

## Recommended environment

Use Python 3.11 or 3.12 on Apple Silicon. Python 3.13 package support is still
 less predictable for this stack.

Example with conda:

```bash
conda env create -f /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/environment-macos.yml
conda activate hf-image-edit-mac
```

## Dependencies

The lab requirements are listed in:

- `/Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/requirements-macos.txt`
- `/Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/environment-macos.yml`

Important tokenizer/runtime dependencies for FLUX-family models:

- `sentencepiece`
- `protobuf`

If your current env is missing them, install them explicitly:

```bash
python -m pip install "sentencepiece>=0.2.0" "protobuf>=4.25.0,<6"
```

## Flux 2 Klein 4B

Text-to-image:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_flux2_klein_base_4b.py \
  --prompt "A cat holding a sign that says hello world" \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/flux_klein_t2i.png
```

Image-to-image with your own reference image:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_flux2_klein_base_4b.py \
  --image /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/test3_pix2pix_exact_visual.png \
  --prompt "Keep the same staircase composition and room layout, but render it more realistically with warm natural materials." \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/flux_klein_edit.png
```

Note:

- `Flux2KleinPipeline` accepts `image=...` for image-conditioned generation, but
  the current Diffusers API does not expose a `strength` parameter on this
  pipeline. The runner keeps `--strength` only for CLI compatibility and ignores
  it.

## FLUX.1-schnell

`FLUX.1-schnell` is a timestep-distilled FLUX model. According to the official
Diffusers docs:

- `guidance_scale` should be `0.0`
- `max_sequence_length` should not exceed `256`
- the model benefits from only `1-4` sampling steps

Source:

- [Diffusers Flux docs](https://huggingface.co/docs/diffusers/en/api/pipelines/flux)
- [FLUX.1-schnell model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

Example:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_flux1_schnell.py \
  --device mps \
  --prompt "A cat holding a sign that says hello world" \
  --height 768 \
  --width 768 \
  --num-inference-steps 4 \
  --guidance-scale 0.0 \
  --max-sequence-length 256 \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/flux1_schnell.png
```

Memory notes for macOS:

- The runner now supports `--offload auto|none|model|sequential`
- On `mps`, `--offload auto` defaults to model CPU offload to avoid moving the
  whole pipeline to MPS at once
- If you still hit MPS OOM, try `--offload sequential --height 512 --width 512`
  and keep `--num-inference-steps` at `2-4`

Safer first try on a constrained Mac:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_flux1_schnell.py \
  --device mps \
  --offload sequential \
  --prompt "A cat holding a sign that says hello world" \
  --height 512 \
  --width 512 \
  --num-inference-steps 2 \
  --guidance-scale 0.0 \
  --max-sequence-length 256 \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/flux1_schnell_small.png
```

## InstructPix2Pix

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_instruct_pix2pix.py \
  --image /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/test3_pix2pix_exact_visual.png \
  --prompt "Keep the same room, staircase, camera angle, table, chairs, and pendant lights. Improve realism only." \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/pix2pix_edit.png
```

Recommended on macOS:

- Start with `512x512`
- Let the runner resize the input image
- On `mps`, the runner uses `float32` for better stability

Example:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_instruct_pix2pix.py \
  --image /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/test3_pix2pix_exact_visual.png \
  --prompt "$(cat /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/prompts/exact_preservation_visual_only.txt)" \
  --height 512 \
  --width 512 \
  --num-inference-steps 20 \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/test3_pix2pix_exact_visual_512.png
```

## Suggested first try on macOS

Run `InstructPix2Pix` first. If that works, then try `FLUX.2-klein-base-4B`.

## Exact Preservation Preset

If you want a fair comparison prompt for both runners, use:

- `/Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/prompts/exact_preservation_visual_only.txt`

Example with `FLUX.2-klein-base-4B`:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_flux2_klein_base_4b.py \
  --image /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/test3_pix2pix_exact_visual.png \
  --prompt "$(cat /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/prompts/exact_preservation_visual_only.txt)" \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/flux_klein_exact_visual.png
```

Example with `InstructPix2Pix`:

```bash
python /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/run_instruct_pix2pix.py \
  --image /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/test3_pix2pix_exact_visual.png \
  --prompt "$(cat /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/prompts/exact_preservation_visual_only.txt)" \
  --output /Users/Yuki/TKNT/experiments/hf_image_edit_macos_lab/outputs/pix2pix_exact_visual.png
```

## Sources

- Diffusers Flux2 docs:
  [https://huggingface.co/docs/diffusers/en/api/pipelines/flux2](https://huggingface.co/docs/diffusers/en/api/pipelines/flux2)
- `FLUX.2-klein-base-4B` model card:
  [https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- `instruct-pix2pix` model card:
  [https://huggingface.co/timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)
- Diffusers MPS guide:
  [https://huggingface.co/docs/diffusers/main/optimization/mps](https://huggingface.co/docs/diffusers/main/optimization/mps)
