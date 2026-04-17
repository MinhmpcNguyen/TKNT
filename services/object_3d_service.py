from __future__ import annotations

import json
import os
import re
import secrets
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import new_session, remove

from adapters.ollama_image_adapter import (
    DEFAULT_OLLAMA_IMAGE_MODEL,
    DEFAULT_OLLAMA_IMAGE_SIZE,
    OllamaGeneratedImage,
    OllamaImageAdapterConfig,
    generate_white_background_image,
    normalize_ollama_endpoint,
)
from adapters.triposr_adapter import TripoSRRunConfig, run_triposr
from config.ollama_config import OllamaConfig

GenerationMode = Literal["prompt", "image"]

DEFAULT_WORK_ROOT = Path(__file__).resolve().parents[1] / "generated" / "object_3d_jobs"
DEFAULT_TRIPOSR_DIR = Path(__file__).resolve().parents[1] / "third_party" / "TripoSR"
PROMPT_IMAGE_MAX_GENERATION_ATTEMPTS = 3


@dataclass(frozen=True)
class Object3DServiceConfig:
    work_root: Path
    triposr_dir: Path
    triposr_python_executable: Path
    triposr_model_name_or_path: str
    triposr_bake_texture_when_available: bool
    triposr_texture_resolution: int
    ollama_endpoint: str | None
    ollama_image_model: str
    ollama_image_size: str

    @classmethod
    def from_env(cls) -> Object3DServiceConfig:
        configured_ollama_endpoint = os.getenv("TKNT_OBJECT_3D_OLLAMA_ENDPOINT")
        ollama_endpoint = normalize_ollama_endpoint(
            configured_ollama_endpoint
        ) or normalize_ollama_endpoint(OllamaConfig.BASE_URL)
        return cls(
            work_root=Path(
                os.getenv("TKNT_OBJECT_3D_WORK_ROOT", str(DEFAULT_WORK_ROOT))
            ).expanduser(),
            triposr_dir=Path(
                os.getenv("TKNT_OBJECT_3D_TRIPOSR_DIR", str(DEFAULT_TRIPOSR_DIR))
            ).expanduser(),
            triposr_python_executable=Path(
                os.getenv("TKNT_OBJECT_3D_TRIPOSR_PYTHON", sys.executable)
            ).expanduser(),
            triposr_model_name_or_path=os.getenv(
                "TKNT_OBJECT_3D_TRIPOSR_MODEL", "stabilityai/TripoSR"
            ),
            triposr_bake_texture_when_available=_read_env_flag(
                "TKNT_OBJECT_3D_TRIPOSR_BAKE_TEXTURE",
                default=True,
            ),
            triposr_texture_resolution=max(
                256,
                _read_env_int(
                    "TKNT_OBJECT_3D_TRIPOSR_TEXTURE_RESOLUTION",
                    default=2048,
                ),
            ),
            ollama_endpoint=ollama_endpoint,
            ollama_image_model=os.getenv(
                "TKNT_OBJECT_3D_OLLAMA_IMAGE_MODEL", DEFAULT_OLLAMA_IMAGE_MODEL
            ),
            ollama_image_size=os.getenv(
                "TKNT_OBJECT_3D_OLLAMA_IMAGE_SIZE", DEFAULT_OLLAMA_IMAGE_SIZE
            ),
        )


@dataclass(frozen=True)
class Object3DJobPaths:
    job_id: str
    root: Path
    status_path: Path
    input_dir: Path
    preprocess_dir: Path
    triposr_dir: Path
    source_image_path: Path
    request_path: Path


@dataclass(frozen=True)
class Object3DPromptRequest:
    prompt: str
    name: str | None = None
    negative_prompt: str | None = None
    keep_debug_artifacts: bool = False
    triposr_device: str = "auto"
    triposr_chunk_size: int | None = None
    triposr_mc_resolution: int = 256
    triposr_mc_threshold: float = 12.0


@dataclass(frozen=True)
class Object3DImageRequest:
    source_filename: str
    name: str | None = None
    keep_debug_artifacts: bool = False
    triposr_device: str = "auto"
    triposr_chunk_size: int | None = None
    triposr_mc_resolution: int = 256
    triposr_mc_threshold: float = 12.0


@dataclass(frozen=True)
class PreprocessResult:
    output_image_path: Path
    debug_paths: dict[str, Path]
    occupancy_ratio: float
    bbox_occupancy_ratio: float


class ObjectTooSmallForReconstructionError(ValueError):
    def __init__(
        self,
        *,
        occupancy_ratio: float,
        min_occupancy_ratio: float,
        bbox_occupancy_ratio: float,
        min_bbox_occupancy_ratio: float,
    ) -> None:
        super().__init__(
            "The detected object is too small for reliable 3D reconstruction. "
            f"Detected occupancy ratio={occupancy_ratio:.3f}; minimum required="
            f"{min_occupancy_ratio:.3f}. "
            f"Detected bbox occupancy ratio={bbox_occupancy_ratio:.3f}; "
            f"minimum required={min_bbox_occupancy_ratio:.3f}."
        )
        self.occupancy_ratio = occupancy_ratio
        self.min_occupancy_ratio = min_occupancy_ratio
        self.bbox_occupancy_ratio = bbox_occupancy_ratio
        self.min_bbox_occupancy_ratio = min_bbox_occupancy_ratio


@dataclass
class TripoSRPreprocessor:
    canvas_size: int = 768
    padding_ratio: float = 0.12
    alpha_threshold: int = 10
    blur_radius: float = 1.0
    contrast_gain: float = 1.08
    sharpness_gain: float = 1.12
    min_object_occupancy_ratio: float = 0.2
    min_object_bbox_occupancy_ratio: float = 0.28
    _session: Any = field(default_factory=new_session, init=False, repr=False)

    def process(
        self,
        *,
        input_path: Path,
        output_path: Path,
        save_debug_dir: Path | None = None,
    ) -> PreprocessResult:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        debug_dir = save_debug_dir
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(input_path).convert("RGBA")
        step1 = self._remove_bg(image)
        if debug_dir is not None:
            step1.save(debug_dir / "01_removed_bg.png")

        step2 = self._clean_alpha_mask(step1)
        if debug_dir is not None:
            step2.save(debug_dir / "02_clean_mask.png")

        occupancy_ratio = self._compute_occupancy_ratio(step2)
        bbox_occupancy_ratio = self._compute_bbox_occupancy_ratio(step2)
        if (
            occupancy_ratio < self.min_object_occupancy_ratio
            and bbox_occupancy_ratio < self.min_object_bbox_occupancy_ratio
        ):
            raise ObjectTooSmallForReconstructionError(
                occupancy_ratio=occupancy_ratio,
                min_occupancy_ratio=self.min_object_occupancy_ratio,
                bbox_occupancy_ratio=bbox_occupancy_ratio,
                min_bbox_occupancy_ratio=self.min_object_bbox_occupancy_ratio,
            )

        step3 = self._crop_with_padding(step2)
        if debug_dir is not None:
            step3.save(debug_dir / "03_cropped.png")

        step4 = self._soften_alpha(step3)
        if debug_dir is not None:
            step4.save(debug_dir / "04_soft_alpha.png")

        step5 = self._center_on_white_square(step4)
        if debug_dir is not None:
            step5.save(debug_dir / "05_square_white.png")

        final_image = self._enhance_rgb(step5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_image.save(output_path)

        debug_paths: dict[str, Path] = {}
        if debug_dir is not None:
            debug_paths = {
                "debug_removed_bg": debug_dir / "01_removed_bg.png",
                "debug_clean_mask": debug_dir / "02_clean_mask.png",
                "debug_cropped": debug_dir / "03_cropped.png",
                "debug_soft_alpha": debug_dir / "04_soft_alpha.png",
                "debug_square_white": debug_dir / "05_square_white.png",
            }

        return PreprocessResult(
            output_image_path=output_path,
            debug_paths=debug_paths,
            occupancy_ratio=occupancy_ratio,
            bbox_occupancy_ratio=bbox_occupancy_ratio,
        )

    def _remove_bg(self, image: Image.Image) -> Image.Image:
        return remove(image.convert("RGBA"), session=self._session).convert("RGBA")

    def _clean_alpha_mask(self, image: Image.Image) -> Image.Image:
        image_array = np.array(image)
        alpha = image_array[:, :, 3]

        _, mask = cv2.threshold(
            alpha,
            self.alpha_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clean_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        else:
            clean_mask = mask

        image_array[:, :, 3] = clean_mask
        return Image.fromarray(image_array, mode="RGBA")

    def _crop_with_padding(self, image: Image.Image) -> Image.Image:
        image_array = np.array(image)
        alpha = image_array[:, :, 3]

        y_coords, x_coords = np.where(alpha > self.alpha_threshold)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return image

        x1 = int(x_coords.min())
        x2 = int(x_coords.max())
        y1 = int(y_coords.min())
        y2 = int(y_coords.max())

        width = x2 - x1 + 1
        height = y2 - y1 + 1

        pad_x = int(width * self.padding_ratio)
        pad_y = int(height * self.padding_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_array.shape[1] - 1, x2 + pad_x)
        y2 = min(image_array.shape[0] - 1, y2 + pad_y)

        return image.crop((x1, y1, x2 + 1, y2 + 1))

    def _soften_alpha(self, image: Image.Image) -> Image.Image:
        red, green, blue, alpha = image.split()
        blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return Image.merge("RGBA", (red, green, blue, blurred_alpha))

    def _center_on_white_square(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image size after preprocessing.")

        scale = min(self.canvas_size / width, self.canvas_size / height) * 0.82
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))

        resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)
        canvas = Image.new(
            "RGBA",
            (self.canvas_size, self.canvas_size),
            (255, 255, 255, 255),
        )
        offset_x = (self.canvas_size - resized_width) // 2
        offset_y = (self.canvas_size - resized_height) // 2
        canvas.paste(resized_image, (offset_x, offset_y), resized_image)
        return canvas

    def _enhance_rgb(self, image: Image.Image) -> Image.Image:
        enhanced_image = image.convert("RGB")
        enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(
            self.contrast_gain
        )
        enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(
            self.sharpness_gain
        )
        return enhanced_image

    def _compute_occupancy_ratio(self, image: Image.Image) -> float:
        alpha = np.array(image)[:, :, 3]
        occupancy = float(np.count_nonzero(alpha > self.alpha_threshold))
        total = float(alpha.shape[0] * alpha.shape[1])
        if total <= 0:
            return 0.0
        return occupancy / total

    def _compute_bbox_occupancy_ratio(self, image: Image.Image) -> float:
        alpha = np.array(image)[:, :, 3]
        y_coords, x_coords = np.where(alpha > self.alpha_threshold)
        total = float(alpha.shape[0] * alpha.shape[1])
        if total <= 0 or len(x_coords) == 0 or len(y_coords) == 0:
            return 0.0

        bbox_width = float(x_coords.max() - x_coords.min() + 1)
        bbox_height = float(y_coords.max() - y_coords.min() + 1)
        return (bbox_width * bbox_height) / total


def create_prompt_job(
    *,
    request: Object3DPromptRequest,
    config: Object3DServiceConfig,
) -> Object3DJobPaths:
    job_paths = _create_job(
        mode="prompt",
        name=request.name,
        config=config,
        source_filename="source.png",
    )
    payload = {
        "mode": "prompt",
        "request": asdict(request),
    }
    job_paths.request_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_status(
        job_paths,
        mode="prompt",
        stage="queued",
        message="Queued 3D generation from prompt.",
        progress_current=0,
        progress_total=3,
    )
    return job_paths


def create_image_job(
    *,
    request: Object3DImageRequest,
    image_bytes: bytes,
    config: Object3DServiceConfig,
) -> Object3DJobPaths:
    job_paths = _create_job(
        mode="image",
        name=request.name,
        config=config,
        source_filename=request.source_filename,
    )
    job_paths.source_image_path.write_bytes(image_bytes)
    payload = {
        "mode": "image",
        "request": asdict(request),
    }
    job_paths.request_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_status(
        job_paths,
        mode="image",
        stage="queued",
        message="Queued 3D generation from uploaded image.",
        progress_current=0,
        progress_total=2,
        artifacts={"source_image": str(job_paths.source_image_path)},
    )
    return job_paths


def run_prompt_job(
    *,
    job_id: str,
    request: Object3DPromptRequest,
    config: Object3DServiceConfig,
) -> None:
    job_paths = get_job_paths(job_id, config)
    try:
        if not config.ollama_endpoint:
            raise RuntimeError(
                "Ollama image endpoint is not configured. Set TKNT_OBJECT_3D_OLLAMA_ENDPOINT "
                "or enable services.ollama.base_url."
            )

        preprocess_result: PreprocessResult | None = None
        artifact_updates: dict[str, str] = {}
        generated: OllamaGeneratedImage | None = None
        prompt_attempt_count = 0
        prompt_variants = _build_prompt_generation_attempts(request.prompt)

        for prompt_attempt_count, prompt_variant in enumerate(prompt_variants, start=1):
            is_retry = prompt_attempt_count > 1
            _write_status(
                job_paths,
                mode="prompt",
                stage="generating_image",
                message=(
                    "Generating a white-background reference image with Ollama."
                    if not is_retry
                    else (
                        "Regenerating the reference image with tighter framing "
                        f"(attempt {prompt_attempt_count}/{len(prompt_variants)})."
                    )
                ),
                progress_current=1,
                progress_total=3,
            )
            generated = generate_white_background_image(
                prompt=prompt_variant,
                output_dir=job_paths.input_dir,
                config=OllamaImageAdapterConfig(
                    endpoint=config.ollama_endpoint,
                    model_name=config.ollama_image_model,
                    output_size=config.ollama_image_size,
                ),
                negative_prompt=request.negative_prompt,
            )
            artifact_updates = _artifacts_from_generated_image(generated)
            _write_status(
                job_paths,
                mode="prompt",
                stage="preprocessing_image",
                message=(
                    "Preprocessing the generated image for TripoSR."
                    if not is_retry
                    else (
                        "Preprocessing the regenerated image for TripoSR "
                        f"(attempt {prompt_attempt_count}/{len(prompt_variants)})."
                    )
                ),
                progress_current=2,
                progress_total=3,
                artifacts=artifact_updates,
            )

            try:
                preprocess_result = _preprocess_for_triposr(
                    source_image_path=generated.image_path,
                    job_paths=job_paths,
                    keep_debug_artifacts=request.keep_debug_artifacts,
                )
                break
            except ObjectTooSmallForReconstructionError as exc:
                if prompt_attempt_count >= len(prompt_variants):
                    raise
                _write_status(
                    job_paths,
                    mode="prompt",
                    stage="preprocessing_image",
                    message=(
                        "The generated object is still too small in frame. "
                        "Retrying with a tighter product-shot prompt."
                    ),
                    progress_current=2,
                    progress_total=3,
                    artifacts=artifact_updates,
                    result={
                        "occupancy_ratio": exc.occupancy_ratio,
                        "bbox_occupancy_ratio": exc.bbox_occupancy_ratio,
                        "prompt_generation_attempt": prompt_attempt_count,
                    },
                )

        if preprocess_result is None or generated is None:
            raise RuntimeError("Prompt image generation did not produce a usable image.")

        _write_status(
            job_paths,
            mode="prompt",
            stage="running_triposr",
            message="Running TripoSR mesh reconstruction.",
            progress_current=3,
            progress_total=3,
            artifacts=_paths_to_string_mapping(
                {
                    **artifact_updates,
                    "preprocessed_image": preprocess_result.output_image_path,
                    **preprocess_result.debug_paths,
                }
            ),
            result={
                "occupancy_ratio": preprocess_result.occupancy_ratio,
                "bbox_occupancy_ratio": preprocess_result.bbox_occupancy_ratio,
                "prompt_generation_attempts": prompt_attempt_count,
            },
        )

        _run_triposr_and_finish(
            job_paths=job_paths,
            preprocess_result=preprocess_result,
            triposr_device=request.triposr_device,
            triposr_chunk_size=request.triposr_chunk_size,
            triposr_mc_resolution=request.triposr_mc_resolution,
            triposr_mc_threshold=request.triposr_mc_threshold,
            config=config,
            mode="prompt",
        )
    except Exception as exc:
        _write_status(
            job_paths,
            mode="prompt",
            stage="error",
            message="3D generation failed.",
            error=str(exc),
        )


def run_image_job(
    *,
    job_id: str,
    request: Object3DImageRequest,
    config: Object3DServiceConfig,
) -> None:
    job_paths = get_job_paths(job_id, config)
    try:
        _write_status(
            job_paths,
            mode="image",
            stage="preprocessing_image",
            message="Preprocessing the uploaded image for TripoSR.",
            progress_current=1,
            progress_total=2,
            artifacts={"source_image": str(job_paths.source_image_path)},
        )
        preprocess_result = _preprocess_for_triposr(
            source_image_path=job_paths.source_image_path,
            job_paths=job_paths,
            keep_debug_artifacts=request.keep_debug_artifacts,
        )
        _write_status(
            job_paths,
            mode="image",
            stage="running_triposr",
            message="Running TripoSR mesh reconstruction.",
            progress_current=2,
            progress_total=2,
            artifacts=_paths_to_string_mapping(
                {
                    "source_image": job_paths.source_image_path,
                    "preprocessed_image": preprocess_result.output_image_path,
                    **preprocess_result.debug_paths,
                }
            ),
            result={
                "occupancy_ratio": preprocess_result.occupancy_ratio,
                "bbox_occupancy_ratio": preprocess_result.bbox_occupancy_ratio,
            },
        )

        _run_triposr_and_finish(
            job_paths=job_paths,
            preprocess_result=preprocess_result,
            triposr_device=request.triposr_device,
            triposr_chunk_size=request.triposr_chunk_size,
            triposr_mc_resolution=request.triposr_mc_resolution,
            triposr_mc_threshold=request.triposr_mc_threshold,
            config=config,
            mode="image",
        )
    except Exception as exc:
        _write_status(
            job_paths,
            mode="image",
            stage="error",
            message="3D generation failed.",
            error=str(exc),
        )


def get_job_paths(job_id: str, config: Object3DServiceConfig) -> Object3DJobPaths:
    root = config.work_root.expanduser().resolve() / job_id
    request_path = root / "request.json"
    if not request_path.exists():
        raise FileNotFoundError(f"3D job not found: {job_id}")
    request_payload = _read_json(request_path)
    source_filename = _resolve_source_filename(request_payload)
    return Object3DJobPaths(
        job_id=job_id,
        root=root,
        status_path=root / "status.json",
        input_dir=root / "inputs",
        preprocess_dir=root / "preprocess_debug",
        triposr_dir=root / "triposr",
        source_image_path=root / "inputs" / source_filename,
        request_path=request_path,
    )


def load_job_status(job_id: str, config: Object3DServiceConfig) -> dict[str, Any]:
    job_paths = get_job_paths(job_id, config)
    if not job_paths.status_path.exists():
        raise FileNotFoundError(f"3D job not found: {job_id}")
    payload = _read_json(job_paths.status_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid status payload for job {job_id}.")
    return payload


def resolve_artifact_path(
    *,
    job_id: str,
    artifact_name: str,
    config: Object3DServiceConfig,
) -> Path:
    job_paths = get_job_paths(job_id, config)
    status_payload = load_job_status(job_id, config)
    artifacts = status_payload.get("artifact_paths")
    if not isinstance(artifacts, dict):
        raise FileNotFoundError(f"No artifacts available for job {job_id}.")

    raw_path = artifacts.get(artifact_name)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise FileNotFoundError(
            f"Artifact {artifact_name!r} not found for job {job_id}."
        )

    resolved_path = Path(raw_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Artifact path no longer exists: {resolved_path}")
    if not resolved_path.is_relative_to(job_paths.root.resolve()):
        raise PermissionError("Artifact path escapes the job workspace.")
    return resolved_path


def _create_job(
    *,
    mode: GenerationMode,
    name: str | None,
    config: Object3DServiceConfig,
    source_filename: str,
) -> Object3DJobPaths:
    config.work_root.mkdir(parents=True, exist_ok=True)
    job_id = _build_job_id(mode=mode, name=name)
    root = config.work_root / job_id
    input_dir = root / "inputs"
    preprocess_dir = root / "preprocess_debug"
    triposr_dir = root / "triposr"
    root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    triposr_dir.mkdir(parents=True, exist_ok=True)
    return Object3DJobPaths(
        job_id=job_id,
        root=root,
        status_path=root / "status.json",
        input_dir=input_dir,
        preprocess_dir=preprocess_dir,
        triposr_dir=triposr_dir,
        source_image_path=input_dir / source_filename,
        request_path=root / "request.json",
    )


def _preprocess_for_triposr(
    *,
    source_image_path: Path,
    job_paths: Object3DJobPaths,
    keep_debug_artifacts: bool,
) -> PreprocessResult:
    debug_dir = job_paths.preprocess_dir if keep_debug_artifacts else None
    preprocessor = TripoSRPreprocessor()
    return preprocessor.process(
        input_path=source_image_path,
        output_path=job_paths.input_dir / "preprocessed_for_triposr.png",
        save_debug_dir=debug_dir,
    )


def _run_triposr_and_finish(
    *,
    job_paths: Object3DJobPaths,
    preprocess_result: PreprocessResult,
    triposr_device: str,
    triposr_chunk_size: int | None,
    triposr_mc_resolution: int,
    triposr_mc_threshold: float,
    config: Object3DServiceConfig,
    mode: GenerationMode,
) -> None:
    triposr_result = run_triposr(
        image_path=preprocess_result.output_image_path,
        output_dir=job_paths.triposr_dir,
        config=TripoSRRunConfig(
            triposr_dir=config.triposr_dir,
            python_executable=config.triposr_python_executable,
            pretrained_model_name_or_path=config.triposr_model_name_or_path,
            device=triposr_device,
            chunk_size=triposr_chunk_size,
            mc_resolution=triposr_mc_resolution,
            mc_threshold=triposr_mc_threshold,
            bake_texture_when_available=config.triposr_bake_texture_when_available,
            texture_resolution=config.triposr_texture_resolution,
        ),
    )
    artifact_paths = {
        "preprocessed_image": preprocess_result.output_image_path,
        **preprocess_result.debug_paths,
        "mesh_glb": triposr_result.mesh_path,
        "mesh_glb_preferred": triposr_result.preferred_mesh_path,
        "triposr_stdout_log": triposr_result.stdout_path,
        "triposr_stderr_log": triposr_result.stderr_path,
    }
    if triposr_result.textured_mesh_path is not None:
        artifact_paths["mesh_glb_textured"] = triposr_result.textured_mesh_path
    if triposr_result.textured_mesh_source_path is not None:
        artifact_paths["mesh_obj_textured"] = triposr_result.textured_mesh_source_path
    if triposr_result.texture_image_path is not None:
        artifact_paths["texture_png"] = triposr_result.texture_image_path
    if triposr_result.texture_stdout_path is not None:
        artifact_paths["triposr_baked_stdout_log"] = triposr_result.texture_stdout_path
    if triposr_result.texture_stderr_path is not None:
        artifact_paths["triposr_baked_stderr_log"] = triposr_result.texture_stderr_path
    if job_paths.source_image_path.exists():
        artifact_paths["source_image"] = job_paths.source_image_path
    result_payload = {
        "mesh_glb_path": str(triposr_result.mesh_path),
        "preferred_mesh_glb_path": str(triposr_result.preferred_mesh_path),
        "occupancy_ratio": preprocess_result.occupancy_ratio,
        "bbox_occupancy_ratio": preprocess_result.bbox_occupancy_ratio,
        "triposr_command": triposr_result.command,
    }
    if triposr_result.texture_command is not None:
        result_payload["triposr_texture_bake_command"] = triposr_result.texture_command
    if triposr_result.texture_bake_error:
        result_payload["texture_bake_error"] = triposr_result.texture_bake_error
    _write_status(
        job_paths,
        mode=mode,
        stage="done",
        message="3D generation finished successfully.",
        progress_current=3 if mode == "prompt" else 2,
        progress_total=3 if mode == "prompt" else 2,
        artifacts=_paths_to_string_mapping(artifact_paths),
        result=result_payload,
    )


def _artifacts_from_generated_image(
    generated: OllamaGeneratedImage,
) -> dict[str, str]:
    return {
        "generated_image": str(generated.image_path),
        "ollama_prompt": str(generated.prompt_path),
        "ollama_negative_prompt": str(generated.negative_prompt_path),
        "ollama_request": str(generated.request_path),
        "ollama_response": str(generated.response_path),
    }


def _build_prompt_generation_attempts(prompt: str) -> list[str]:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return [normalized_prompt]

    return [
        normalized_prompt,
        (
            f"{normalized_prompt}\n\n"
            "Framing requirements: use a close-up centered studio product shot. "
            "The single object must fill most of the image with only small white margins. "
            "Avoid any zoomed-out composition. "
            "Use a gentle three-quarter front camera angle so the object shows visible side depth. "
            "Reveal natural material detail, tactile surface texture, and soft shape-defining shadows."
        ),
        (
            f"{normalized_prompt}\n\n"
            "Framing requirements: move the camera closer until the object fills about 85% "
            "of the frame while remaining fully visible. Keep a pure white background and "
            "leave almost no empty white space around the subject. "
            "Use a photorealistic catalog hero shot from a three-quarter view, not a flat head-on cutout. "
            "Favor rich surface detail, believable local color, and clear depth cues across the full body."
        ),
    ][:PROMPT_IMAGE_MAX_GENERATION_ATTEMPTS]


def _write_status(
    job_paths: Object3DJobPaths,
    *,
    mode: GenerationMode,
    stage: str,
    message: str,
    progress_current: int | None = None,
    progress_total: int | None = None,
    error: str | None = None,
    artifacts: dict[str, str] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    existing = (
        _read_json(job_paths.status_path) if job_paths.status_path.exists() else {}
    )
    existing_artifacts = existing.get("artifact_paths")
    artifact_paths = (
        dict(existing_artifacts) if isinstance(existing_artifacts, dict) else {}
    )
    if artifacts is not None:
        artifact_paths.update(artifacts)

    actions = existing.get("actions")
    action_history = (
        [item for item in actions if isinstance(item, dict)]
        if isinstance(actions, list)
        else []
    )
    updated_at_utc = _now_utc_iso()
    action_history.append(
        {
            "stage": stage,
            "message": message,
            "updated_at_utc": updated_at_utc,
            "progress_current": progress_current,
            "progress_total": progress_total,
            "error": error,
        }
    )

    payload: dict[str, Any] = {
        "job_id": job_paths.job_id,
        "job_dir": str(job_paths.root),
        "mode": mode,
        "stage": stage,
        "updated_at_utc": updated_at_utc,
        "message": message,
        "error": error,
        "progress_current": progress_current,
        "progress_total": progress_total,
        "artifact_paths": artifact_paths,
        "actions": action_history,
    }
    if result is not None:
        payload["result"] = result
    elif isinstance(existing.get("result"), dict):
        payload["result"] = existing["result"]

    job_paths.status_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source_filename(payload: Any) -> str:
    if isinstance(payload, dict):
        request_payload = payload.get("request")
        if isinstance(request_payload, dict):
            source_filename = request_payload.get("source_filename")
            if isinstance(source_filename, str) and source_filename.strip():
                return source_filename
    return "source.png"


def _paths_to_string_mapping(paths: dict[str, Path | str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, value in paths.items():
        mapped[key] = str(value)
    return mapped


def _read_env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _read_env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _build_job_id(*, mode: GenerationMode, name: str | None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(name or mode)
    suffix = secrets.token_hex(3)
    return f"obj3d_{mode}_{timestamp}_{slug}_{suffix}"


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized[:40] or "job"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
