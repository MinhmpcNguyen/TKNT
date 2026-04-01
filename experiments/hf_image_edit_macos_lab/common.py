from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    torch_dtype_name: str


def build_shared_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--prompt", required=True, type=str, help="Prompt text.")
    parser.add_argument("--image", type=Path, help="Optional input image for editing.")
    parser.add_argument("--output", required=True, type=Path, help="Output image path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Execution device. Default: auto-detect.",
    )
    return parser


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_optional_image(path: Path | None) -> None:
    if path is not None and not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")


def select_runtime(
    *,
    torch_module: object,
    requested_device: str,
) -> RuntimeConfig:
    torch = torch_module

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device

    if device == "cuda":
        torch_dtype_name = "bfloat16"
    elif device == "mps":
        torch_dtype_name = "float16"
    else:
        torch_dtype_name = "float32"

    return RuntimeConfig(device=device, torch_dtype_name=torch_dtype_name)


def resolve_torch_dtype(torch_module: object, dtype_name: str) -> object:
    torch = torch_module
    return getattr(torch, dtype_name)


def seed_everything(torch_module: object, seed: int) -> None:
    torch = torch_module
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
