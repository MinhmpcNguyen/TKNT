# TripoSR on macOS

## 1. Create an environment

```bash
cd /Users/Yuki/TKNT
python -m venv .conda_envs/triposr-mac
source .conda_envs/triposr-mac/bin/activate
python -m pip install --upgrade pip setuptools wheel cmake ninja
```

If native builds fail on macOS, install Xcode Command Line Tools first:

```bash
xcode-select --install
```

## 2. Install PyTorch

For Apple Silicon / macOS:

```bash
pip install torch torchvision
```

## 3. Clone TripoSR

```bash
cd /Users/Yuki/TKNT
git clone https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR
```

## 4. Install TripoSR dependencies

This follows the official TripoSR requirements, with `xatlas` and `gradio` kept optional unless you need texture baking or the Gradio UI.

```bash
pip install \
  omegaconf==2.3.0 \
  Pillow==10.1.0 \
  einops==0.7.0 \
  transformers==4.35.0 \
  trimesh==4.0.5 \
  rembg \
  huggingface-hub \
  "imageio[ffmpeg]" \
  moderngl==5.10.0
pip install git+https://github.com/tatsy/torchmcubes.git
```

Optional extras for the official `run.py` / Gradio UI / texture baking:

```bash
pip install xatlas==0.0.9 gradio
```

## 5. Run the macOS wrapper

```bash
cd /Users/Yuki/TKNT
python experiments/test_triposr_mac.py \
  /Users/Yuki/TKNT/experiments/chair.png \
  --triposr-dir /Users/Yuki/TKNT/third_party/TripoSR \
  --output-dir /Users/Yuki/TKNT/experiments/triposr_output
```

## 6. View the generated GLB

```bash
python experiments/view_glb.py \
  --input /Users/Yuki/TKNT/experiments/triposr_output/0/mesh.glb \
  --open
```
