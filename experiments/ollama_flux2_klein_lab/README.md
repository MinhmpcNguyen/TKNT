# Ollama FLUX.2 Klein Lab

This folder is a self-contained playground for testing `x/flux2-klein:latest`
locally in two ways:

1. `reference image + prompt`
2. `final layout JSON + prompt`

The scripts use only the Python standard library, so you can run them with
`python3` as long as your local Ollama server is available at
`http://localhost:11434`.

## Files

- `case_1_reference_image.py`
  Tries to find a working local path for `image + prompt -> generated image`.
  It probes:
  - `POST /v1/images/edits` (OpenAI-style, experimental)
  - `POST /api/generate` with an `images` array (probe path)
- `case_1_exact_preservation.py`
  Runs a stricter reference-image test that tries to preserve the same scene,
  framing, object count, and layout as closely as possible.
- `case_2_layout_json.py`
  Uses the official `POST /v1/images/generations` endpoint and injects your
  final layout JSON into the prompt.
- `sample_layout.json`
  A sample final layout JSON you can replace with your own output.
- `prompts/exact_preservation_prompt.txt`
  The default stricter prompt used by `case_1_exact_preservation.py`.
- `common.py`
  Shared HTTP, prompt, and image-decoding utilities.

## Before Running

Start Ollama if it is not already running:

```bash
ollama serve
```

## Case 1: Reference Image + Prompt

```bash
python3 experiments/ollama_flux2_klein_lab/case_1_reference_image.py \
  --image /absolute/path/to/reference.png \
  --prompt "Keep the same room layout and camera angle, but render it as a warm Japandi interior with oak wood, linen textures, and soft daylight."
```

What happens:

- The script saves raw responses for each attempted endpoint.
- If an endpoint returns image bytes or base64 image data, it writes the result
  into `experiments/ollama_flux2_klein_lab/outputs/`.
- If no image is returned, the script exits with a clear error and keeps the
  raw probe artifacts for inspection.

## Case 2: Final Layout JSON + Prompt

```bash
python3 experiments/ollama_flux2_klein_lab/case_2_layout_json.py \
  --layout-json experiments/ollama_flux2_klein_lab/sample_layout.json \
  --prompt "Create a clean top-down interior render of this living room. Preserve the object count, relative spacing, circulation, and overall footprint. Use warm neutral materials, natural daylight, and a realistic floor texture."
```

This flow uses the officially documented image-generation route and is the more
stable of the two.

## Exact Preservation Check

```bash
python3 experiments/ollama_flux2_klein_lab/case_1_exact_preservation.py \
  --image /absolute/path/to/reference.png
```

This flow is intended for a stricter validation pass:

- It uses a tighter default prompt focused on scene preservation.
- It writes the exact prompt used into `outputs_exact/...prompt.txt`.
- It writes a run report into `outputs_exact/...report.json`.
- By default it uses only `/api/generate`, because that is the route that has
  returned images most reliably in this lab.

If you still want to probe the experimental edits route too:

```bash
python3 experiments/ollama_flux2_klein_lab/case_1_exact_preservation.py \
  --image /absolute/path/to/reference.png \
  --try-openai-edits
```

If you want to try image-edit mode directly:

```bash
python3 experiments/ollama_flux2_klein_lab/case_1_exact_preservation.py \
  --image /absolute/path/to/reference.png \
  --mode edits-only \
  --edits-image-field both
```

Useful edit-mode flags:

- `--mode edits-only`
- `--mode edits-then-api-generate`
- `--edits-image-field image`
- `--edits-image-field image[]`
- `--edits-image-field both`
- `--mask /absolute/path/to/mask.png`

## Optional Flags

Both scripts support these useful flags:

- `--model x/flux2-klein:latest`
- `--endpoint http://localhost:11434`
- `--output /absolute/or/relative/output/path.png`
- `--prompt-file /path/to/prompt.txt`

## Notes

- `case_1_reference_image.py` is intentionally a probe, because Ollama publicly
  documents image generation for FLUX.2 Klein but does not clearly document one
  stable reference-image editing endpoint yet.
- `case_2_layout_json.py` works well for testing whether your final layout JSON
  can steer composition and styling through the prompt alone.
