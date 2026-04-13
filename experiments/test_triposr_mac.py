from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    fallback_device: str | None


@dataclass(frozen=True)
class TripoSRModules:
    TSR: Any
    rembg: Any
    remove_background: Any
    resize_foreground: Any
    save_video: Any


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "triposr_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TripoSR on macOS with MPS->CPU fallback and export a GLB mesh.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--triposr-dir",
        type=Path,
        default=None,
        help="Path to a cloned VAST-AI-Research/TripoSR repository.",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="stabilityai/TripoSR",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto-detect.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Renderer chunk size. Smaller uses less memory but is slower.",
    )
    parser.add_argument(
        "--mc-resolution",
        type=int,
        default=256,
        help="Marching cubes resolution.",
    )
    parser.add_argument(
        "--foreground-ratio",
        type=float,
        default=0.85,
        help="Foreground ratio used after background removal.",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Skip automatic background removal.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render a preview video.",
    )
    parser.add_argument(
        "--render-views",
        type=int,
        default=30,
        help="Number of views for the optional preview video.",
    )
    return parser.parse_args()


def resolve_triposr_dir(requested_dir: Path | None) -> Path:
    candidate_paths = [
        requested_dir,
        Path(os.environ["TRIPOSR_DIR"]).expanduser()
        if "TRIPOSR_DIR" in os.environ
        else None,
        Path.cwd() / "TripoSR",
        Path(__file__).resolve().parents[1] / "TripoSR",
        Path(__file__).resolve().parents[1] / "third_party" / "TripoSR",
    ]

    for candidate_path in candidate_paths:
        if candidate_path is None:
            continue
        resolved_path = candidate_path.expanduser().resolve()
        if (resolved_path / "run.py").exists() and (resolved_path / "tsr").is_dir():
            return resolved_path

    raise FileNotFoundError(
        "Could not find a TripoSR checkout. Pass --triposr-dir or set TRIPOSR_DIR."
    )


def load_triposr_modules(triposr_dir: Path) -> TripoSRModules:
    sys.path.insert(0, str(triposr_dir))

    tsr_system = import_module("tsr.system")
    tsr_utils = import_module("tsr.utils")
    rembg_module = import_module("rembg")

    return TripoSRModules(
        TSR=tsr_system.TSR,
        rembg=rembg_module,
        remove_background=tsr_utils.remove_background,
        resize_foreground=tsr_utils.resize_foreground,
        save_video=tsr_utils.save_video,
    )


def pick_runtime(requested_device: str) -> RuntimeConfig:
    import torch

    if requested_device == "cpu":
        return RuntimeConfig(device="cpu", fallback_device=None)

    if requested_device == "mps":
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return RuntimeConfig(device="mps", fallback_device="cpu")
        raise RuntimeError(
            "MPS was requested, but it is not available in this PyTorch build."
        )

    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return RuntimeConfig(device="mps", fallback_device="cpu")

    return RuntimeConfig(device="cpu", fallback_device=None)


def prepare_image(
    *,
    image_path: Path,
    modules: TripoSRModules,
    remove_bg: bool,
    foreground_ratio: float,
) -> Image.Image:
    source_image = Image.open(image_path)

    if remove_bg:
        rembg_session = modules.rembg.new_session()
        image_rgba = modules.remove_background(source_image, rembg_session)
        image_rgba = modules.resize_foreground(image_rgba, foreground_ratio)
        image_array = np.array(image_rgba).astype(np.float32) / 255.0
        image_array = (
            image_array[:, :, :3] * image_array[:, :, 3:4]
            + (1.0 - image_array[:, :, 3:4]) * 0.5
        )
        return Image.fromarray((image_array * 255.0).astype(np.uint8))

    return source_image.convert("RGB")


def run_inference(
    *,
    modules: TripoSRModules,
    model_name_or_path: str,
    processed_image: Image.Image,
    output_dir: Path,
    runtime: RuntimeConfig,
    chunk_size: int,
    mc_resolution: int,
    render: bool,
    render_views: int,
) -> Path:
    import torch

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if runtime.device == "mps" else "0"

    model = modules.TSR.from_pretrained(
        model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(runtime.device)

    with torch.no_grad():
        scene_codes = model([processed_image], device=runtime.device)

    if render:
        render_images = model.render(
            scene_codes, n_views=render_views, return_type="pil"
        )
        for frame_index, render_image in enumerate(render_images[0]):
            render_image.save(output_dir / f"render_{frame_index:03d}.png")
        modules.save_video(render_images[0], str(output_dir / "render.mp4"), fps=30)

    meshes = model.extract_mesh(
        scene_codes, has_vertex_color=True, resolution=mc_resolution
    )
    mesh_path = output_dir / "mesh.glb"
    meshes[0].export(mesh_path)
    return mesh_path


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    triposr_dir = resolve_triposr_dir(args.triposr_dir)
    runtime = pick_runtime(args.device)
    modules = load_triposr_modules(triposr_dir)
    processed_image = prepare_image(
        image_path=input_path,
        modules=modules,
        remove_bg=not args.no_remove_bg,
        foreground_ratio=args.foreground_ratio,
    )

    sample_output_dir = output_dir / "0"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    processed_image.save(sample_output_dir / "input.png")

    print(f"[INFO] TripoSR repo: {triposr_dir}")
    print(f"[INFO] Runtime: {runtime.device}")
    print(f"[INFO] Input:   {input_path}")
    print(f"[INFO] Output:  {sample_output_dir}")

    started_at = time.perf_counter()
    try:
        mesh_path = run_inference(
            modules=modules,
            model_name_or_path=args.pretrained_model_name_or_path,
            processed_image=processed_image,
            output_dir=sample_output_dir,
            runtime=runtime,
            chunk_size=args.chunk_size,
            mc_resolution=args.mc_resolution,
            render=args.render,
            render_views=args.render_views,
        )
    except Exception as exc:
        if runtime.device == "mps" and runtime.fallback_device == "cpu":
            print(f"[WARN] MPS failed: {exc}")
            print("[WARN] Retrying on CPU...")
            fallback_runtime = RuntimeConfig(device="cpu", fallback_device=None)
            mesh_path = run_inference(
                modules=modules,
                model_name_or_path=args.pretrained_model_name_or_path,
                processed_image=processed_image,
                output_dir=sample_output_dir,
                runtime=fallback_runtime,
                chunk_size=args.chunk_size,
                mc_resolution=args.mc_resolution,
                render=args.render,
                render_views=args.render_views,
            )
        else:
            raise

    total_elapsed = time.perf_counter() - started_at
    print(f"[DONE] Mesh: {mesh_path}")
    print(f"[DONE] Total time: {total_elapsed:.2f}s")
    print(
        "[TIP] View it with: "
        f"python /Users/Yuki/TKNT/experiments/view_glb.py --input {mesh_path} --open"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
