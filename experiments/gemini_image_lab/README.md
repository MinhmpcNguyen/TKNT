# Gemini Image Lab

This folder contains a standalone runner for Gemini image generation and
image editing using the same Gemini service configuration style as the repo's
`GeminiClient`.

It supports:

- `gemini-3.1-flash-image-preview`
- `gemini-2.5-flash-image`

As of April 1, 2026, Google's official image-generation docs list:

- `gemini-3.1-flash-image-preview` as the "Gemini 3.1 Flash Image Preview"
- `gemini-2.5-flash-image` as the "Gemini 2.5 Flash Image" model

Sources:

- [Gemini image generation docs](https://ai.google.dev/gemini-api/docs/image-generation)
- [GenerateContent API reference](https://ai.google.dev/api/generate-content)

## Runner

- `run_gemini_image.py`

The runner:

- reads Gemini API settings from `GeminiConfig` first
- falls back to `GEMINI_API_KEY` and `GEMINI_BASE_URL` if needed
- supports text-to-image and text-and-image-to-image
- saves image output, text output, and raw JSON response

## Prompt preset

- `prompts/exact_preservation_visual_only.txt`

This is the same style of prompt used for the Hugging Face comparison runs:
it constrains camera, lighting, color, framing, and composition without naming
specific furniture or room objects.

## Example: Gemini 3.1 Flash Image Preview

```bash
python /Users/Yuki/TKNT/experiments/gemini_image_lab/run_gemini_image.py \
  --model gemini-3.1-flash-image-preview \
  --image /Users/Yuki/TKNT/experiments/ollama_flux2_klein_lab/test3.png \
  --prompt-file /Users/Yuki/TKNT/experiments/gemini_image_lab/prompts/exact_preservation_visual_only.txt \
  --output /Users/Yuki/TKNT/experiments/gemini_image_lab/outputs/test3_gemini_31_preview.png
```

## Example: Gemini 2.5 Flash Image

```bash
python /Users/Yuki/TKNT/experiments/gemini_image_lab/run_gemini_image.py \
  --model gemini-2.5-flash-image \
  --image /Users/Yuki/TKNT/experiments/ollama_flux2_klein_lab/test3.png \
  --prompt-file /Users/Yuki/TKNT/experiments/gemini_image_lab/prompts/exact_preservation_visual_only.txt \
  --output /Users/Yuki/TKNT/experiments/gemini_image_lab/outputs/test3_gemini_25_flash_image.png
```

## Useful options

- `--aspect-ratio 16:9`
- `--image-size 2K`
- `--image-only`

## Output files

If the model returns both text and image parts, the runner writes:

- the first image to the exact `--output` path if possible
- additional images next to it with numeric suffixes
- text to `*.txt`
- full raw JSON to `*.response.json`
