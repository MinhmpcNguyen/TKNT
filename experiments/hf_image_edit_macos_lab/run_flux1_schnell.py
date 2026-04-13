from __future__ import annotations

import argparse

from common import ensure_parent_dir, resolve_torch_dtype, seed_everything, select_runtime

MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLUX.1-schnell on macOS-friendly settings.",
    )
    parser.add_argument("--prompt", required=True, type=str, help="Prompt text.")
    parser.add_argument("--output", required=True, type=str, help="Output image path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=768, help="Output height.")
    parser.add_argument("--width", type=int, default=768, help="Output width.")
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Execution device. Default: auto-detect.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale. Official docs recommend 0.0 for FLUX.1-schnell.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="Number of denoising steps. Official docs recommend 1-4 for schnell.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Maximum sequence length. Official docs cap this at 256 for schnell.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Model identifier or local model path.",
    )
    parser.add_argument(
        "--offload",
        choices=("auto", "none", "model", "sequential"),
        default="auto",
        help=(
            "CPU offload mode. On macOS, 'auto' uses model offload to reduce MPS memory pressure."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from diffusers import FluxPipeline
    from pathlib import Path

    output_path = Path(args.output)
    ensure_parent_dir(output_path)

    runtime = select_runtime(torch_module=torch, requested_device=args.device)
    torch_dtype = resolve_torch_dtype(torch_module=torch, dtype_name=runtime.torch_dtype_name)

    print(f"Loading {args.model_id} on device={runtime.device} dtype={runtime.torch_dtype_name}")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    try:
        configure_pipeline(pipe=pipe, device=runtime.device, offload_mode=args.offload)
    except RuntimeError as exc:
        if "MPS backend out of memory" in str(exc):
            print("MPS ran out of memory while preparing the pipeline.")
            print("Try again with:")
            print("  --offload sequential")
            print("  --height 512 --width 512")
            print("  --num-inference-steps 2")
            return 2
        raise

    seed_everything(torch_module=torch, seed=args.seed)
    generator_device = "cpu" if runtime.device == "mps" else runtime.device
    generator = torch.Generator(generator_device).manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        height=args.height,
        width=args.width,
        generator=generator,
    ).images[0]
    image.save(output_path)
    print(f"Saved {output_path}")
    return 0


def configure_pipeline(*, pipe: object, device: str, offload_mode: str) -> None:
    resolved_offload_mode = resolve_offload_mode(device=device, requested_mode=offload_mode)

    if resolved_offload_mode == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif resolved_offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()


def resolve_offload_mode(*, device: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    if device in {"cuda", "mps"}:
        return "model"

    return "none"


if __name__ == "__main__":
    raise SystemExit(main())
