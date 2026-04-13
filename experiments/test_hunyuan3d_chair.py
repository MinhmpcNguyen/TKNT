from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors

DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "chair.png"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "hunyuan3d_outputs"

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
DTypeName = Literal["auto", "float32", "float16", "bfloat16"]
FlashVDMMode = Literal["auto", "on", "off"]
MCAlgoName = Literal["auto", "mc", "dmc"]
PresetName = Literal["fast", "balanced", "quality"]


@dataclass(frozen=True)
class GenerationPreset:
    model: str
    subfolder: str
    num_inference_steps: int
    guidance_scale: float
    octree_resolution: int
    num_chunks: int
    enable_flashvdm: bool


@dataclass(frozen=True)
class ResolvedRunConfig:
    input_path: Path
    outdir: Path
    name: str
    model: str
    subfolder: str
    device: str
    torch_dtype: torch.dtype
    num_inference_steps: int
    guidance_scale: float
    octree_resolution: int
    num_chunks: int
    enable_flashvdm: bool
    mc_algo: str | None
    seed: int | None


PRESETS: dict[PresetName, GenerationPreset] = {
    "fast": GenerationPreset(
        model="tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        num_inference_steps=5,
        guidance_scale=5.0,
        octree_resolution=128,
        num_chunks=8000,
        enable_flashvdm=True,
    ),
    "balanced": GenerationPreset(
        model="tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        num_inference_steps=8,
        guidance_scale=5.0,
        octree_resolution=192,
        num_chunks=8000,
        enable_flashvdm=True,
    ),
    "quality": GenerationPreset(
        model="tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        num_inference_steps=50,
        guidance_scale=5.0,
        octree_resolution=384,
        num_chunks=8000,
        enable_flashvdm=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3D chair mesh with Hunyuan3D and export OBJ/GLB.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where OBJ and GLB exports will be written.",
    )
    parser.add_argument(
        "--name",
        default="chair",
        help="Base filename for exported mesh files.",
    )
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "quality"),
        default="fast",
        help="Runtime preset. Default: fast.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Override the model subfolder inside the pretrained repo.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto-detect.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype. Default: choose a safe dtype for the selected device.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override diffusion steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override classifier-free guidance scale.",
    )
    parser.add_argument(
        "--octree-resolution",
        type=int,
        default=None,
        help="Override mesh extraction octree resolution.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Override the number of decoder chunks.",
    )
    parser.add_argument(
        "--flashvdm",
        choices=("auto", "on", "off"),
        default="auto",
        help="Enable FlashVDM mesh decoding. Default: follow the selected preset.",
    )
    parser.add_argument(
        "--mc-algo",
        choices=("auto", "mc", "dmc"),
        default="auto",
        help="Surface extraction algorithm. Default: pick a safe option automatically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible results.",
    )
    parser.add_argument(
        "--disable-pbar",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without loading the model.",
    )
    return parser.parse_args()


def resolve_device(requested_device: DeviceName) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build does not have CUDA available."
        )

    if requested_device == "mps":
        if (
            getattr(torch.backends, "mps", None) is None
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested, but this PyTorch build does not have MPS available."
            )

    return requested_device


def resolve_dtype(*, requested_dtype: DTypeName, device: str) -> torch.dtype:
    if requested_dtype == "auto":
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32

    return getattr(torch, requested_dtype)


def resolve_flashvdm(*, requested_mode: FlashVDMMode, preset: GenerationPreset) -> bool:
    if requested_mode == "auto":
        return preset.enable_flashvdm
    return requested_mode == "on"


def resolve_mc_algo(
    *,
    requested_algo: MCAlgoName,
    device: str,
    enable_flashvdm: bool,
) -> str | None:
    if requested_algo != "auto":
        return requested_algo
    if enable_flashvdm and device in {"cpu", "mps"}:
        return "mc"
    return None


def resolve_generator(*, device: str, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None

    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(generator_device).manual_seed(seed)


def prepare_scheduler_for_device(
    *,
    scheduler: object,
    device: str,
) -> None:
    if device != "mps":
        return

    # Hunyuan3D turbo schedulers may initialize timestep tensors as float64,
    # which MPS cannot move onto device memory.
    for attribute_name in ("sigmas", "timesteps"):
        attribute_value = getattr(scheduler, attribute_name, None)
        if (
            isinstance(attribute_value, torch.Tensor)
            and attribute_value.is_floating_point()
        ):
            setattr(scheduler, attribute_name, attribute_value.to(dtype=torch.float32))


def configure_surface_extractor(
    *,
    pipeline: Hunyuan3DDiTFlowMatchingPipeline,
    mc_algo: str | None,
) -> None:
    if mc_algo is None:
        return

    pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]()


def build_run_config(args: argparse.Namespace) -> ResolvedRunConfig:
    preset = PRESETS[args.preset]
    input_path = args.input.resolve()
    outdir = args.outdir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(requested_dtype=args.dtype, device=device)
    enable_flashvdm = resolve_flashvdm(requested_mode=args.flashvdm, preset=preset)
    mc_algo = resolve_mc_algo(
        requested_algo=args.mc_algo,
        device=device,
        enable_flashvdm=enable_flashvdm,
    )

    return ResolvedRunConfig(
        input_path=input_path,
        outdir=outdir,
        name=args.name,
        model=args.model if args.model is not None else preset.model,
        subfolder=args.subfolder if args.subfolder is not None else preset.subfolder,
        device=device,
        torch_dtype=torch_dtype,
        num_inference_steps=(
            args.steps if args.steps is not None else preset.num_inference_steps
        ),
        guidance_scale=(
            args.guidance_scale
            if args.guidance_scale is not None
            else preset.guidance_scale
        ),
        octree_resolution=(
            args.octree_resolution
            if args.octree_resolution is not None
            else preset.octree_resolution
        ),
        num_chunks=args.num_chunks
        if args.num_chunks is not None
        else preset.num_chunks,
        enable_flashvdm=enable_flashvdm,
        mc_algo=mc_algo,
        seed=args.seed,
    )


def print_run_config(config: ResolvedRunConfig) -> None:
    print("Resolved config:")
    print(f"  input={config.input_path}")
    print(f"  outdir={config.outdir}")
    print(f"  name={config.name}")
    print(f"  model={config.model}")
    print(f"  subfolder={config.subfolder}")
    print(f"  device={config.device}")
    print(f"  dtype={config.torch_dtype}")
    print(f"  steps={config.num_inference_steps}")
    print(f"  guidance_scale={config.guidance_scale}")
    print(f"  octree_resolution={config.octree_resolution}")
    print(f"  num_chunks={config.num_chunks}")
    print(f"  flashvdm={config.enable_flashvdm}")
    print(f"  mc_algo={config.mc_algo}")
    print(f"  seed={config.seed}")


def main() -> int:
    args = parse_args()
    config = build_run_config(args)

    print_run_config(config)
    if args.dry_run:
        return 0

    if config.device == "cpu":
        print(
            "No CUDA or MPS backend is available. Falling back to CPU; generation may be slow."
        )

    config.outdir.mkdir(parents=True, exist_ok=True)
    obj_path = config.outdir / f"{config.name}.obj"
    glb_path = config.outdir / f"{config.name}.glb"
    generator = resolve_generator(device=config.device, seed=config.seed)

    total_started_at = time.perf_counter()

    load_started_at = time.perf_counter()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        config.model,
        subfolder=config.subfolder,
        device=config.device,
        dtype=config.torch_dtype,
    )
    prepare_scheduler_for_device(scheduler=pipeline.scheduler, device=config.device)
    if config.enable_flashvdm:
        pipeline.enable_flashvdm(mc_algo=config.mc_algo or "mc")
    configure_surface_extractor(pipeline=pipeline, mc_algo=config.mc_algo)
    load_finished_at = time.perf_counter()

    generation_started_at = time.perf_counter()
    mesh = pipeline(
        image=str(config.input_path),
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        octree_resolution=config.octree_resolution,
        num_chunks=config.num_chunks,
        generator=generator,
        enable_pbar=not args.disable_pbar,
    )[0]
    generation_finished_at = time.perf_counter()

    export_started_at = time.perf_counter()
    mesh.export(obj_path)
    mesh.export(glb_path)
    export_finished_at = time.perf_counter()

    total_finished_at = time.perf_counter()

    print(f"Saved OBJ: {obj_path}")
    print(f"Saved GLB: {glb_path}")
    print(f"Load time: {load_finished_at - load_started_at:.2f}s")
    print(f"Generation time: {generation_finished_at - generation_started_at:.2f}s")
    print(f"Export time: {export_finished_at - export_started_at:.2f}s")
    print(f"Total time: {total_finished_at - total_started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
