from __future__ import annotations

from pathlib import Path
from shutil import copy2

from PIL import Image, ImageDraw


ROOT = Path("/Users/Yuki/TKNT/Thesis_LaTeX")
OVERLEAF_ROOT = Path("/Users/Yuki/TKNT/roomflow_overleaf_latex_source-5-")
BACKUP = ROOT / "figures_backup_small_annotations_20260627"
OVERLEAF_BACKUP = OVERLEAF_ROOT / "figures_backup_small_annotations_20260627"


def backup(path: Path, backup_root: Path, root: Path) -> None:
    target = backup_root / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        copy2(path, target)


def fill_box(image: Image.Image, box: tuple[int, int, int, int], sample: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    color = image.getpixel(sample)
    draw.rectangle(box, fill=color)


def clean_image(relative_path: str, boxes: list[tuple[tuple[int, int, int, int], tuple[int, int]]]) -> None:
    local_path = ROOT / relative_path
    overleaf_path = OVERLEAF_ROOT / relative_path

    backup(local_path, BACKUP, ROOT)
    if overleaf_path.exists():
        backup(overleaf_path, OVERLEAF_BACKUP, OVERLEAF_ROOT)

    with Image.open(local_path) as image:
        clean = image.convert("RGBA")
        for box, sample in boxes:
            fill_box(clean, box, sample)
        clean.save(local_path)

    if overleaf_path.exists():
        overleaf_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(local_path, overleaf_path)


def main() -> None:
    clean_image(
        "figures/chapter5/edit_case_example.png",
        [
            ((0, 70, 1900, 104), (20, 80)),
            ((0, 655, 1900, 735), (20, 700)),
        ],
    )
    clean_image(
        "figures/chapter5/edit_output_example.png",
        [
            ((0, 64, 1300, 96), (20, 80)),
            ((0, 758, 1300, 815), (20, 790)),
        ],
    )

    for relative_path in [
        "figures/chapter4/case_comparisons/simple_living_01_comparison.png",
        "figures/chapter4/case_comparisons/living_l_shape_comparison.png",
        "figures/chapter4/case_comparisons/bedroom_chamfer_comparison.png",
    ]:
        clean_image(
            relative_path,
            [
                ((0, 68, 1680, 116), (20, 82)),
            ],
        )

    clean_image(
        "figures/chapter4/image_flow_cases/create_variant_inputs.png",
        [
            ((0, 63, 1640, 112), (20, 82)),
            ((42, 198, 540, 226), (50, 212)),
            ((574, 198, 1074, 226), (584, 212)),
            ((1108, 198, 1608, 226), (1118, 212)),
        ],
    )
    clean_image(
        "figures/chapter4/image_flow_cases/create_variant_outputs.png",
        [
            ((0, 63, 1640, 112), (20, 82)),
            ((42, 198, 540, 226), (50, 212)),
            ((574, 198, 1074, 226), (584, 212)),
            ((1108, 198, 1608, 226), (1118, 212)),
            ((42, 615, 540, 645), (50, 630)),
            ((574, 615, 1074, 645), (584, 630)),
            ((1108, 615, 1608, 645), (1118, 630)),
        ],
    )
    clean_image(
        "figures/chapter4/figure_14_preset_fidelity.png",
        [
            ((0, 80, 1450, 130), (20, 100)),
        ],
    )
    clean_image(
        "figures/chapter4/image_flow_cases/edit_variant_comparison.png",
        [
            ((0, 68, 1776, 106), (20, 82)),
        ],
    )
    clean_image(
        "figures/chapter4/figure_06_2d.png",
        [
            ((0, 66, 1300, 102), (20, 82)),
        ],
    )


if __name__ == "__main__":
    main()
