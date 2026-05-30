from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from .types import (
    AlignMode,
    CaseConfig,
    JSONValue,
    ManifestConfig,
    MetricDirection,
    MetricResult,
    MetricStatus,
    VariantConfig,
)

FloatArray: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]
BoolMask: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class PairImages:
    reference: Image.Image
    candidate: Image.Image
    details: Mapping[str, JSONValue]


def evaluate_variant_images(
    *,
    manifest: ManifestConfig,
    case: CaseConfig,
    variant: VariantConfig,
    output_image: Path,
    output_dir: Path,
) -> list[MetricResult]:
    results: list[MetricResult] = []
    metrics = set(manifest.metrics)
    if "blank" in metrics:
        results.append(
            compute_blank_check(
                output_image,
                thresholds=manifest.thresholds,
            )
        )

    if case.reference_image is not None:
        results.extend(
            compute_pair_metrics(
                prefix="reference",
                reference_path=case.reference_image,
                candidate_path=output_image,
                mask_path=None,
                selected_region="all",
                output_dir=output_dir,
                metrics=metrics,
                thresholds=manifest.thresholds,
                visual_delta_threshold=manifest.visual_delta_threshold,
                align_mode=manifest.align_mode,
                case_id=case.id,
                variant_id=variant.id,
            )
        )

    if case.source_image is not None and case.mask_image is not None:
        results.extend(
            compute_pair_metrics(
                prefix="outside_mask_preservation",
                reference_path=case.source_image,
                candidate_path=output_image,
                mask_path=case.mask_image,
                selected_region="outside_mask",
                output_dir=output_dir,
                metrics=metrics,
                thresholds=manifest.thresholds,
                visual_delta_threshold=manifest.visual_delta_threshold,
                align_mode=manifest.align_mode,
                case_id=case.id,
                variant_id=variant.id,
            )
        )

    if case.target_image is not None:
        target_region = "inside_mask" if case.mask_image is not None else "all"
        results.extend(
            compute_pair_metrics(
                prefix="target",
                reference_path=case.target_image,
                candidate_path=output_image,
                mask_path=case.mask_image,
                selected_region=target_region,
                output_dir=output_dir,
                metrics=metrics,
                thresholds=manifest.thresholds,
                visual_delta_threshold=manifest.visual_delta_threshold,
                align_mode=manifest.align_mode,
                case_id=case.id,
                variant_id=variant.id,
            )
        )

    return results


def compute_blank_check(
    image_path: Path,
    *,
    thresholds: Mapping[str, float],
) -> MetricResult:
    try:
        image = _open_rgba(image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return MetricResult(
            name="blank",
            status="failed",
            details={"reason": f"image_decode_failed: {exc}"},
        )

    array = np.asarray(image, dtype=np.uint8)
    rgb = array[:, :, :3].astype(np.float32)
    alpha = array[:, :, 3]
    width, height = image.size
    file_bytes = image_path.stat().st_size
    pixel_stddev = float(np.std(rgb))
    background = _corner_background_rgb(rgb)
    background_delta = np.max(np.abs(rgb - background), axis=2)
    non_background_ratio = float(np.mean(background_delta > 3.0))
    alpha_coverage = float(np.mean(alpha > 0))
    min_stddev = thresholds.get("min_pixel_stddev", 3.0)
    min_non_background = thresholds.get("min_non_background_ratio", 0.001)
    min_file_bytes = thresholds.get("min_file_bytes", 128.0)
    min_alpha_coverage = thresholds.get("min_alpha_coverage", 0.001)
    passed = (
        pixel_stddev >= min_stddev
        and non_background_ratio >= min_non_background
        and file_bytes >= min_file_bytes
        and alpha_coverage >= min_alpha_coverage
    )
    return MetricResult(
        name="blank",
        status="passed" if passed else "failed",
        value=pixel_stddev,
        threshold=min_stddev,
        direction="min",
        details={
            "width": width,
            "height": height,
            "file_bytes": file_bytes,
            "pixel_stddev": pixel_stddev,
            "non_background_ratio": non_background_ratio,
            "alpha_coverage": alpha_coverage,
            "min_non_background_ratio": min_non_background,
            "min_file_bytes": min_file_bytes,
            "min_alpha_coverage": min_alpha_coverage,
        },
    )


def compute_pair_metrics(
    *,
    prefix: str,
    reference_path: Path,
    candidate_path: Path,
    mask_path: Path | None,
    selected_region: str,
    output_dir: Path,
    metrics: set[str],
    thresholds: Mapping[str, float],
    visual_delta_threshold: int,
    align_mode: AlignMode,
    case_id: str,
    variant_id: str,
) -> list[MetricResult]:
    try:
        pair = _prepare_pair(
            reference_path=reference_path,
            candidate_path=candidate_path,
            align_mode=align_mode,
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        return [
            MetricResult(
                name=f"{prefix}_alignment",
                status="failed",
                details={"reason": str(exc)},
            )
        ]

    selected_mask = _load_selected_mask(
        mask_path=mask_path,
        size=pair.reference.size,
        selected_region=selected_region,
    )
    reference_rgb = _rgb_array(pair.reference)
    candidate_rgb = _rgb_array(pair.candidate)
    results: list[MetricResult] = []

    if "visual_diff" in metrics or "visual" in metrics:
        results.append(
            compute_visual_diff(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                output_dir=output_dir,
                case_id=case_id,
                variant_id=variant_id,
                thresholds=thresholds,
                visual_delta_threshold=visual_delta_threshold,
                extra_details=pair.details,
            )
        )

    if "psnr" in metrics:
        results.append(
            compute_psnr(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                thresholds=thresholds,
                extra_details=pair.details,
            )
        )

    if "ssim" in metrics:
        results.append(
            compute_ssim(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                thresholds=thresholds,
                extra_details=pair.details,
            )
        )

    if "nrmse" in metrics:
        results.append(
            compute_nrmse(
                prefix=prefix,
                reference_rgb=reference_rgb,
                candidate_rgb=candidate_rgb,
                selected_mask=selected_mask,
                thresholds=thresholds,
                extra_details=pair.details,
            )
        )

    return results


def compute_visual_diff(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    output_dir: Path,
    case_id: str,
    variant_id: str,
    thresholds: Mapping[str, float],
    visual_delta_threshold: int,
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    selected = _selected_mask_or_all(selected_mask, reference_rgb.shape[:2])
    if not bool(np.any(selected)):
        return _skipped(f"{prefix}_visual_diff", "selected_region_is_empty")

    diff = np.abs(reference_rgb.astype(np.int16) - candidate_rgb.astype(np.int16))
    channel_max = np.max(diff, axis=2)
    selected_channel_max = channel_max[selected]
    selected_diff = diff[selected]
    diff_pixel_ratio = float(np.mean(selected_channel_max > visual_delta_threshold))
    mean_abs_error = float(np.mean(selected_diff))
    max_channel_delta = int(np.max(selected_channel_max))
    diff_path = _write_diff_image(
        diff=diff,
        selected_mask=selected,
        output_dir=output_dir,
        case_id=case_id,
        variant_id=variant_id,
        prefix=prefix,
    )
    max_ratio = _threshold(thresholds, prefix, "max_diff_pixel_ratio")
    max_mean_abs_error = _threshold(thresholds, prefix, "max_mean_abs_error")
    passed = True
    if max_ratio is not None:
        passed = passed and diff_pixel_ratio <= max_ratio
    if max_mean_abs_error is not None:
        passed = passed and mean_abs_error <= max_mean_abs_error
    status: MetricStatus = "passed" if passed else "failed"
    if max_ratio is None and max_mean_abs_error is None:
        status = "warning"
    details = {
        **dict(extra_details),
        "diff_pixel_ratio": diff_pixel_ratio,
        "mean_abs_error": mean_abs_error,
        "max_channel_delta": max_channel_delta,
        "visual_delta_threshold": visual_delta_threshold,
        "diff_image": str(diff_path),
    }
    if max_mean_abs_error is not None:
        details["max_mean_abs_error"] = max_mean_abs_error
    return MetricResult(
        name=f"{prefix}_visual_diff",
        status=status,
        value=diff_pixel_ratio,
        threshold=max_ratio,
        direction="max",
        details=details,
    )


def compute_psnr(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    thresholds: Mapping[str, float],
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    selected_reference, selected_candidate = _selected_pixels(
        reference_rgb,
        candidate_rgb,
        selected_mask,
    )
    if selected_reference.size == 0:
        return _skipped(f"{prefix}_psnr", "selected_region_is_empty")
    mse = float(np.mean((selected_reference - selected_candidate) ** 2))
    perfect_match = mse == 0.0
    value = 100.0 if perfect_match else 20.0 * math.log10(255.0 / math.sqrt(mse))
    threshold = _threshold(thresholds, prefix, "min_psnr")
    status = _status_for_threshold(value=value, threshold=threshold, direction="min")
    return MetricResult(
        name=f"{prefix}_psnr",
        status=status,
        value=value,
        threshold=threshold,
        direction="min",
        details={**dict(extra_details), "mse": mse, "perfect_match": perfect_match},
    )


def compute_nrmse(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    thresholds: Mapping[str, float],
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    selected_reference, selected_candidate = _selected_pixels(
        reference_rgb,
        candidate_rgb,
        selected_mask,
    )
    if selected_reference.size == 0:
        return _skipped(f"{prefix}_nrmse", "selected_region_is_empty")
    rmse = float(np.sqrt(np.mean((selected_reference - selected_candidate) ** 2)))
    value = rmse / 255.0
    threshold = _threshold(thresholds, prefix, "max_nrmse")
    status = _status_for_threshold(value=value, threshold=threshold, direction="max")
    return MetricResult(
        name=f"{prefix}_nrmse",
        status=status,
        value=value,
        threshold=threshold,
        direction="max",
        details={**dict(extra_details), "rmse": rmse, "normalization": "255"},
    )


def compute_ssim(
    *,
    prefix: str,
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
    thresholds: Mapping[str, float],
    extra_details: Mapping[str, JSONValue],
) -> MetricResult:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return _skipped(f"{prefix}_ssim", "scikit-image is not installed")

    min_side = min(reference_rgb.shape[0], reference_rgb.shape[1])
    if min_side < 3:
        return _skipped(f"{prefix}_ssim", "image_too_small")
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if selected_mask is not None:
        ignored = np.logical_not(selected_mask)
        candidate_for_metric = candidate_rgb.copy()
        candidate_for_metric[ignored] = reference_rgb[ignored]
    else:
        candidate_for_metric = candidate_rgb
    value = float(
        structural_similarity(
            reference_rgb,
            candidate_for_metric,
            channel_axis=2,
            data_range=255,
            win_size=win_size,
        )
    )
    threshold = _threshold(thresholds, prefix, "min_ssim")
    status = _status_for_threshold(value=value, threshold=threshold, direction="min")
    return MetricResult(
        name=f"{prefix}_ssim",
        status=status,
        value=value,
        threshold=threshold,
        direction="min",
        details={**dict(extra_details), "win_size": win_size},
    )


def summarize_human_ratings(
    *,
    ratings_csv: Path,
    artifact_fail_threshold: float = 3.0,
) -> dict[str, JSONValue]:
    groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with ratings_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            variant_id = (row.get("variant") or row.get("variant_id") or "").strip()
            if not case_id or not variant_id:
                continue
            groups[(case_id, variant_id)].append(
                {
                    "score_instruction": _optional_float(row.get("score_instruction")),
                    "score_quality": _optional_float(row.get("score_quality")),
                    "score_preservation": _optional_float(
                        row.get("score_preservation")
                    ),
                    "artifact_severity": _optional_float(row.get("artifact_severity")),
                }
            )
    output: dict[str, JSONValue] = {}
    for (case_id, variant_id), rows in groups.items():
        key = f"{case_id}/{variant_id}"
        output[key] = {
            "case_id": case_id,
            "variant_id": variant_id,
            "rater_count": len(rows),
            "mean_instruction": _mean(row["score_instruction"] for row in rows),
            "mean_quality": _mean(row["score_quality"] for row in rows),
            "mean_preservation": _mean(row["score_preservation"] for row in rows),
            "mean_artifact_severity": _mean(row["artifact_severity"] for row in rows),
            "artifact_fail_rate": _rate(
                row["artifact_severity"] >= artifact_fail_threshold for row in rows
            ),
        }
    return output


def _prepare_pair(
    *,
    reference_path: Path,
    candidate_path: Path,
    align_mode: AlignMode,
) -> PairImages:
    reference = _open_rgba(reference_path).convert("RGB")
    candidate = _open_rgba(candidate_path).convert("RGB")
    details: dict[str, JSONValue] = {
        "reference_image": str(reference_path),
        "candidate_image": str(candidate_path),
        "reference_width": reference.width,
        "reference_height": reference.height,
        "candidate_width": candidate.width,
        "candidate_height": candidate.height,
        "candidate_resized": False,
    }
    if reference.size != candidate.size:
        if align_mode == "strict":
            raise ValueError(
                f"Image dimensions differ: reference={reference.size}, "
                f"candidate={candidate.size}."
            )
        candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
        details["candidate_resized"] = True
        details["aligned_width"] = reference.width
        details["aligned_height"] = reference.height
    return PairImages(reference=reference, candidate=candidate, details=details)


def _open_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _rgb_array(image: Image.Image) -> UInt8Array:
    return cast(UInt8Array, np.asarray(image.convert("RGB"), dtype=np.uint8))


def _load_selected_mask(
    *,
    mask_path: Path | None,
    size: tuple[int, int],
    selected_region: str,
) -> BoolMask | None:
    if mask_path is None or selected_region == "all":
        return None
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    if selected_region == "inside_mask":
        return cast(BoolMask, mask_array)
    if selected_region == "outside_mask":
        return cast(BoolMask, np.logical_not(mask_array))
    raise ValueError(f"Unknown selected region: {selected_region}")


def _selected_mask_or_all(mask: BoolMask | None, shape: tuple[int, int]) -> BoolMask:
    if mask is not None:
        return mask
    return cast(BoolMask, np.ones(shape, dtype=np.bool_))


def _selected_pixels(
    reference_rgb: UInt8Array,
    candidate_rgb: UInt8Array,
    selected_mask: BoolMask | None,
) -> tuple[FloatArray, FloatArray]:
    selected = _selected_mask_or_all(selected_mask, reference_rgb.shape[:2])
    return (
        cast(FloatArray, reference_rgb[selected].astype(np.float32)),
        cast(FloatArray, candidate_rgb[selected].astype(np.float32)),
    )


def _write_diff_image(
    *,
    diff: NDArray[np.int16],
    selected_mask: BoolMask,
    output_dir: Path,
    case_id: str,
    variant_id: str,
    prefix: str,
) -> Path:
    visible_diff = diff.copy()
    visible_diff[np.logical_not(selected_mask)] = 0
    enhanced = np.clip(visible_diff * 4, 0, 255).astype(np.uint8)
    path = output_dir / case_id / variant_id / f"{prefix}_diff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced, mode="RGB").save(path)
    return path


def _corner_background_rgb(rgb: FloatArray) -> FloatArray:
    height, width = rgb.shape[:2]
    patch = max(1, min(height, width, 8))
    samples = np.concatenate(
        [
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, width - patch :].reshape(-1, 3),
            rgb[height - patch :, :patch].reshape(-1, 3),
            rgb[height - patch :, width - patch :].reshape(-1, 3),
        ],
        axis=0,
    )
    return cast(FloatArray, np.median(samples, axis=0))


def _threshold(
    thresholds: Mapping[str, float],
    prefix: str,
    key: str,
) -> float | None:
    for candidate in (f"{prefix}.{key}", f"{prefix}_{key}", key):
        if candidate in thresholds:
            return thresholds[candidate]
    return None


def _status_for_threshold(
    *,
    value: float,
    threshold: float | None,
    direction: MetricDirection,
) -> MetricStatus:
    if threshold is None or direction == "none":
        return "warning"
    if direction == "min":
        return "passed" if value >= threshold else "failed"
    return "passed" if value <= threshold else "failed"


def _skipped(name: str, reason: str) -> MetricResult:
    return MetricResult(name=name, status="skipped", details={"reason": reason})


def _optional_float(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _mean(values: Iterable[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _rate(values: Iterable[bool]) -> float | None:
    clean = list(values)
    if not clean:
        return None
    return sum(1 for value in clean if value) / len(clean)
