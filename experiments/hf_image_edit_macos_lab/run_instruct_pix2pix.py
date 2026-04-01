from __future__ import annotations

import argparse

from common import build_shared_parser, ensure_parent_dir, resolve_torch_dtype
from common import seed_everything, select_runtime, validate_optional_image

MODEL_ID = "timbrooks/instruct-pix2pix"


def parse_args() -> argparse.Namespace:
    parser = build_shared_parser("Run InstructPix2Pix on macOS-friendly settings.")
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="How strongly to preserve the input image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="How strongly to follow the text prompt.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Model identifier or local model path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from PIL import Image

    if args.image is None:
        raise ValueError("--image is required for InstructPix2Pix.")

    validate_optional_image(args.image)
    ensure_parent_dir(args.output)

    runtime = select_runtime(torch_module=torch, requested_device=args.device)
    torch_dtype = resolve_torch_dtype(torch_module=torch, dtype_name=runtime.torch_dtype_name)

    print(f"Loading {args.model_id} on device={runtime.device} dtype={runtime.torch_dtype_name}")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    configure_pipeline(pipe=pipe, device=runtime.device)

    seed_everything(torch_module=torch, seed=args.seed)

    input_image = Image.open(args.image).convert("RGB")
    image = pipe(
        prompt=args.prompt,
        image=input_image,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.num_inference_steps,
    ).images[0]
    image.save(args.output)
    print(f"Saved {args.output}")
    return 0


def configure_pipeline(*, pipe: object, device: str) -> None:
    pipe.to(device)
    pipe.enable_attention_slicing()


if __name__ == "__main__":
    raise SystemExit(main())
