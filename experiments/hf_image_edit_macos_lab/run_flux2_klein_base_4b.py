from __future__ import annotations

import argparse
import warnings

from common import build_shared_parser, ensure_parent_dir, resolve_torch_dtype
from common import seed_everything, select_runtime, validate_optional_image

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"


def parse_args() -> argparse.Namespace:
    parser = build_shared_parser("Run FLUX.2-klein-base-4B on macOS-friendly settings.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Kept for CLI compatibility. Flux2KleinPipeline does not expose a strength argument.",
    )
    parser.add_argument(
        "--caption-upsample-temperature",
        type=float,
        help="Optional Flux2 caption upsampling temperature.",
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
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    validate_optional_image(args.image)
    ensure_parent_dir(args.output)

    runtime = select_runtime(torch_module=torch, requested_device=args.device)
    torch_dtype = resolve_torch_dtype(torch_module=torch, dtype_name=runtime.torch_dtype_name)

    print(f"Loading {args.model_id} on device={runtime.device} dtype={runtime.torch_dtype_name}")
    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    configure_pipeline(pipe=pipe, device=runtime.device)

    seed_everything(torch_module=torch, seed=args.seed)

    call_kwargs: dict[str, object] = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
    }
    if args.caption_upsample_temperature is not None:
        call_kwargs["caption_upsample_temperature"] = args.caption_upsample_temperature
    if args.image is not None:
        call_kwargs["image"] = Image.open(args.image).convert("RGB")
        if args.strength != 0.75:
            warnings.warn(
                "Flux2KleinPipeline does not support a `strength` argument in the current "
                "Diffusers API, so --strength is ignored.",
                stacklevel=2,
            )

    image = pipe(**call_kwargs).images[0]
    image.save(args.output)
    print(f"Saved {args.output}")
    return 0


def configure_pipeline(*, pipe: object, device: str) -> None:
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        return

    pipe.to(device)
    if device == "mps":
        pipe.enable_attention_slicing()


if __name__ == "__main__":
    raise SystemExit(main())
